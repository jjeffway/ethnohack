# load libraries
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

# silence RDKit C++ warnings
RDLogger.DisableLog("rdApp.warning")

# Fingerprint settings are only needed when a model was trained on fingerprint
# bits. Descriptor-based models do not need them, so a missing constant is not
# an error here.
try:
    from ethnohack.constants import FP_SIZE, MORGAN_RADIUS, RANDOM_STATE
except ImportError:  # pragma: no cover
    from ethnohack.constants import RANDOM_STATE

    FP_SIZE, MORGAN_RADIUS = 2048, 2

# cache explainers (building one for a large forest takes a while)
_explainer_cache = {}


# ============================================================
# FEATURE LAYOUT
# Two kinds of model are supported:
#   fingerprint models  features named fp_0 ... fp_<FP_SIZE-1>, optionally
#                       followed by extra descriptor columns
#   descriptor models   RDKit descriptor columns only (MolWt, TPSA, fr_phenol…)
# Structure highlighting works for both: fingerprint bits are drawn from their
# Morgan environments, fr_* descriptors from their SMARTS patterns.
# ============================================================
def _load_fragment_smarts():
    """
    name -> SMARTS for RDKit's fr_* fragment descriptors, read from the data
    file RDKit ships. Returns {} if the file layout ever changes, in which
    case fragment highlighting is skipped rather than failing.
    """
    try:
        import rdkit.Chem.Fragments as frag_mod

        patterns = {}
        with open(frag_mod.defaultPatternFileName) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0].replace("=", "_").replace("-", "_")
                patt = Chem.MolFromSmarts(parts[2].strip())
                if patt is not None and patt.GetNumAtoms():
                    patterns[name] = patt
        return patterns
    except Exception:
        return {}


_FRAGMENT_PATTERNS = _load_fragment_smarts()


def fingerprint_positions(feature_names):
    """Morgan bit id -> position in the feature list (empty for descriptor models)."""
    return {
        int(n[3:]): i
        for i, n in enumerate(feature_names)
        if n.startswith("fp_") and n[3:].isdigit()
    }


def has_fingerprints(feature_names):
    """True when the model was trained on Morgan fingerprint bits."""
    return bool(fingerprint_positions(feature_names))


def feature_kind(name):
    """What a feature column is, used for labelling and highlighting."""
    base = name.split(" ")[0]          # readable names may carry a hint suffix
    if base.startswith("fp_") and base[3:].isdigit():
        return "fingerprint bit"
    if base in _FRAGMENT_PATTERNS:
        return "fragment count"
    return "descriptor"


# check that the model's feature layout matches what this module assumes
def check_feature_layout(feature_names, model=None, bundle=None):
    names = list(feature_names)

    # feature count must match what the model expects
    if model is not None and len(names) != model.n_features_in_:
        raise ValueError(
            f"{len(names)} feature names but model expects {model.n_features_in_} features"
        )

    if has_fingerprints(names):
        positions = fingerprint_positions(names)

        # bit drawing maps feature index -> Morgan bit id, so the bits must sit
        # at the front of the feature list, in bit order
        expected = [f"fp_{i}" for i in range(len(positions))]
        if [n.split(" ")[0] for n in names[: len(positions)]] != expected:
            raise ValueError(
                f"The first {len(positions)} features are not fp_0 ... "
                f"fp_{len(positions) - 1}, so bit drawing would be wrong"
            )

        # fingerprint settings saved with the model must match the ones used here
        if bundle is not None and "fp_size" in bundle:
            if bundle["fp_size"] != FP_SIZE or bundle.get("radius") != MORGAN_RADIUS:
                raise ValueError(
                    f"Model trained with fp_size={bundle['fp_size']}, "
                    f"radius={bundle.get('radius')} but constants say "
                    f"{FP_SIZE}, {MORGAN_RADIUS}"
                )

        print(f"Fingerprint model: {len(positions)} bits + "
              f"{len(names) - len(positions)} extra features")
        return "fingerprint"

    # descriptor model: nothing to validate about ordering, since each column
    # stands on its own and carries its meaning in its name
    kinds = pd.Series([feature_kind(n) for n in names]).value_counts()
    print(f"Descriptor model: {len(names)} features "
          f"({kinds.get('fragment count', 0)} fragment counts, "
          f"{kinds.get('descriptor', 0)} other descriptors)")
    if not _FRAGMENT_PATTERNS:
        print("NOTE: fragment SMARTS unavailable, structure highlighting disabled")
    return "descriptor"


# add substructure hints to the fp_* feature names (single pass over the molecules)
def readable_feature_names(feature_names, smiles_list):
    names = list(feature_names)
    positions = fingerprint_positions(names)

    # descriptor models need no hints: MolWt and fr_phenol already say what
    # they mean, and unlike hashed bits each name has exactly one meaning
    if not positions:
        return names

    seen = set()
    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        if mol is None:
            continue

        for bit_id, envs in get_bit_info(mol).items():
            if bit_id in seen or bit_id not in positions:
                continue

            # name the bit after one substructure that switches it on
            atom, radius = envs[0]
            env = (
                Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom) if radius else None
            )
            frag = (
                Chem.MolFragmentToSmiles(mol, atomsToUse=[atom])
                if env is None
                else Chem.MolToSmiles(Chem.PathToSubmol(mol, env))
            )
            if frag:
                names[positions[bit_id]] = f"{names[positions[bit_id]]} {frag}"
            seen.add(bit_id)

        # stop early once every bit has a hint
        if len(seen) == len(positions):
            break

    return names


# build a TreeExplainer once per model and reuse it
def _get_explainer(model):
    # keep a reference to the model so its id cannot be reused by another object
    if id(model) not in _explainer_cache:
        _explainer_cache[id(model)] = (model, shap.TreeExplainer(model))
    return _explainer_cache[id(model)][1]


# convert SMILES input -> RDKit molecule (None for non-string or invalid input)
def _mol_from_smiles(smiles):
    # NaN/None from CSV columns would raise inside MolFromSmiles, so check type first
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


# get the model input vector for one molecule
def _features_for(smiles, featurizer=None):
    # no featurizer: only valid for models trained on fingerprint bits alone
    if featurizer is None:
        return smiles_to_ecfp4(smiles)

    # model-specific featurizer (e.g. cox_model_api.featurize) -> the exact
    # columns used for training, in training order
    if _mol_from_smiles(smiles) is None:
        return None
    return np.asarray(featurizer([smiles]), dtype=float)[0]


# function to convert SMILES str -> fingerprint array
def smiles_to_ecfp4(smiles):
    # accept only string-like SMILES input
    if not isinstance(smiles, str):
        return None

    # use RDkit MolFromSmiles function to convert SMILES -> RDkit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # handle invalid SMILES -> return None so callers can handle it cleanly
    if mol is None:
        return None

    # extract Morgan Fingerprint (ECFP4) -> use RDkit fingerprint generator
    # NOTE: radius, size and bit (not count) type must match the training run
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=FP_SIZE
    )
    fp = generator.GetFingerprint(mol)

    # convert RDKit ExplicitBitVect -> NumPy array
    fp_array = np.zeros((len(fp),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, fp_array)

    return fp_array


# return Morgan bit information for one RDKit molecule
def get_bit_info(mol):
    # return empty map when molecule is invalid
    if mol is None:
        return {}

    # generate fingerprint generator with additional output for bit mapping
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=FP_SIZE
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    generator.GetFingerprint(mol, additionalOutput=ao)

    # return map from fingerprint bit to atom/radius environments
    return ao.GetBitInfoMap()


# ============================================================
# SHAP PLUMBING (unchanged)
# ============================================================
def _resolve_class_index(model, target_class):
    # if model exposes classes_, map label to position first
    # labels take priority so labels like 1/2 are not mistaken for positions
    if hasattr(model, "classes_"):
        classes = np.asarray(model.classes_)
        matches = np.where(classes == target_class)[0]
        if len(matches) > 0:
            return int(matches[0])

    # if target_class is already a valid integer position, keep it
    if isinstance(target_class, (int, np.integer)):
        return int(target_class)

    return int(target_class)


def _standardize_shap_values(shap_values, model):
    # TreeExplainer returns arrays or lists depending on model and shap version
    if isinstance(shap_values, list):
        # list of class arrays -> stack to (n_samples, n_features, n_classes)
        return np.stack(shap_values, axis=-1)

    shap_array = np.asarray(shap_values)

    # some shap versions return (n_classes, n_samples, n_features)
    if shap_array.ndim == 3:
        n_classes = len(getattr(model, "classes_", []))
        if n_classes == 0:
            n_classes = shap_array.shape[-1]

        # class axis already last -> keep (checked first so a batch of n_classes
        # molecules is not transposed by mistake)
        if shap_array.shape[-1] == n_classes:
            return shap_array

        if shap_array.shape[0] == n_classes:
            return np.transpose(shap_array, (1, 2, 0))

        return shap_array

    # binary/single-output models may return (n_samples, n_features)
    return shap_array


def _get_expected_value(explainer, class_idx, model=None):
    expected = explainer.expected_value
    expected_array = np.asarray(expected)

    # scalar expected value is used for binary/single-output models
    if np.isscalar(expected) or expected_array.size == 1:
        value = float(expected_array.ravel()[0])

        # single-output binary models explain the positive class only, so flip
        # the sign when the negative class (position 0) is requested
        if (
            model is not None
            and len(getattr(model, "classes_", [])) == 2
            and class_idx == 0
        ):
            value = -value

        return value

    return float(expected_array[class_idx])


def _select_class_shap(shap_values, model, class_idx):
    # slice class column (class shapley values)
    if shap_values.ndim == 3:
        return shap_values[:, :, class_idx]

    shap_class = np.asarray(shap_values)

    # single-output binary models explain the positive class only
    if len(getattr(model, "classes_", [])) == 2 and class_idx == 0:
        shap_class = -shap_class

    return shap_class


def get_class_expected_value(model, target_class):
    explainer = _get_explainer(model)
    class_idx = _resolve_class_index(model, target_class)
    return _get_expected_value(explainer, class_idx, model)


# ============================================================
# PER-MOLECULE SHAP
# ============================================================
def get_shap_features(smiles, model, target_class, chembl_id=None, featurizer=None):
    """
    SHAP values for one molecule.
    Returns (features, id, explainer, shap_values, sorted_features) where
    sorted_features is [(feature_index, shap_value), ...] strongest first.
    """
    # get model input features for the molecule
    mol_explain = _features_for(smiles, featurizer)

    # stop cleanly when SMILES cannot be parsed
    if mol_explain is None:
        print(f"Invalid SMILES string: {smiles}")
        return None

    # pass ID directly because dataset/model are defined outside this module
    id_explain = str(chembl_id) if chembl_id is not None else None

    # set up the explainer using shap TreeExplainer for the model (cached)
    explainer = _get_explainer(model)

    # reshape to (1, n_features) for SHAP
    shap_values = explainer.shap_values(
        mol_explain.reshape(1, -1), check_additivity=False
    )

    # standardize SHAP output across SHAP versions and model types
    shap_values = _standardize_shap_values(shap_values, model)
    class_idx = _resolve_class_index(model, target_class)
    shap_class = _select_class_shap(shap_values, model, class_idx)[0]

    # sort features by absolute SHAP value
    sorted_features = sorted(
        enumerate(shap_class), key=lambda x: abs(x[1]), reverse=True
    )

    return mol_explain, id_explain, explainer, shap_class, sorted_features


# kept so existing notebooks calling get_shap_bits keep working
get_shap_bits = get_shap_features


# ============================================================
# STRUCTURE HIGHLIGHTING
# ============================================================
def _draw_highlighted(mol, atom_colors, bond_colors, width=790, height=600):
    """Render one molecule with the given atoms and bonds coloured."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()      # keeps non-highlighted atoms clean
    options.continuousHighlight = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    IPython.display.display(IPython.display.SVG(drawer.GetDrawingText()))


# RGB palette: red pushes toward active, blue toward inactive, gray for zero
RED = (1.0, 0.2, 0.2)
BLUE = (0.2, 0.4, 1.0)
GRAY = (0.6, 0.6, 0.6)


def _color_for(shap_val):
    return RED if shap_val > 0 else BLUE if shap_val < 0 else GRAY


def visualize_mol_shap(
    smiles,
    model,
    target_class,
    n_top=2,
    chembl_id=None,
    featurizer=None,
    feature_names=None,
):
    """
    Draw the molecule with the strongest contributing features highlighted.

    Fingerprint models highlight the Morgan environments of the top bits.
    Descriptor models highlight the SMARTS matches of the top fr_* fragment
    counts, which is the closest equivalent; features like MolWt describe the
    whole molecule and cannot be localised, so they are reported in text only.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return

    result = get_shap_features(
        smiles, model, target_class, chembl_id=chembl_id, featurizer=featurizer
    )
    if result is None:
        return

    mol_explain, id_explain, explainer, shap_class, sorted_features = result

    if id_explain is not None:
        print(f"Molecule ChEMBL ID: {id_explain}")

    names = list(feature_names) if feature_names is not None else None
    fingerprint_model = names is not None and has_fingerprints(names)
    if names is None:
        fingerprint_model = True   # no names given: assume the old bit layout

    atom_colors, bond_colors = {}, {}

    if fingerprint_model:
        bit_info = get_bit_info(mol)

        # only bits actually present in this molecule can be drawn
        present = [(i, v) for i, v in sorted_features if i in bit_info]
        selected = present[:n_top]
        if not selected:
            print("No present bits found to highlight.")
            return

        print(f"=== Overlaying top {len(selected)} present features ===")
        for idx, val in selected:
            label = names[idx] if names else f"Bit {idx}"
            print(f"{label:40s} | SHAP: {val:+.6f}")

        # colour weakest first so the strongest wins where environments overlap
        for idx, val in reversed(selected):
            color = _color_for(val)
            for central_atom, radius in bit_info[idx]:
                atom_colors[central_atom] = color
                if radius > 0:
                    for bond_idx in Chem.FindAtomEnvironmentOfRadiusN(
                        mol, radius, central_atom
                    ):
                        bond_colors[bond_idx] = color
                        bond = mol.GetBondWithIdx(bond_idx)
                        atom_colors[bond.GetBeginAtomIdx()] = color
                        atom_colors[bond.GetEndAtomIdx()] = color
    else:
        # descriptor model: localise the fr_* fragment counts
        localisable = []
        for idx, val in sorted_features:
            base = names[idx].split(" ")[0]
            patt = _FRAGMENT_PATTERNS.get(base)
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt)
            if matches:
                localisable.append((idx, val, matches))
            if len(localisable) >= n_top:
                break

        print(f"=== Top {n_top} contributing features ===")
        for idx, val in sorted_features[:n_top]:
            print(f"{names[idx]:28s} | value {mol_explain[idx]:>8.3f} "
                  f"| SHAP {val:+.6f} | {feature_kind(names[idx])}")

        if not localisable:
            print("\nNone of this molecule's fragment descriptors contribute, "
                  "so there is nothing to draw. Whole-molecule features such "
                  "as MolWt cannot be localised.")
            return

        # the drawn features are the strongest ones that map to a substructure,
        # which are not always the strongest overall
        print(f"\n=== Drawn on the structure (strongest localisable) ===")
        for idx, val, matches in localisable:
            print(f"{names[idx]:28s} | SHAP {val:+.6f} | {len(matches)}x | "
                  f"{'red (active)' if val > 0 else 'blue (inactive)'}")

        # colour weakest first so the strongest wins where matches overlap
        for idx, val, matches in reversed(localisable):
            color = _color_for(val)
            for match in matches:
                for atom_idx in match:
                    atom_colors[atom_idx] = color
                for bond in mol.GetBonds():
                    if (bond.GetBeginAtomIdx() in match
                            and bond.GetEndAtomIdx() in match):
                        bond_colors[bond.GetIdx()] = color

    _draw_highlighted(mol, atom_colors, bond_colors)


# create waterfall plot for contributing features
def waterfall_plot(shap_class, expected_value, mol_explain, feature_names=None):
    if feature_names is None:
        sample_exp = shap.Explanation(
            values=shap_class, base_values=expected_value, data=mol_explain
        )
    else:
        sample_exp = shap.Explanation(
            values=shap_class,
            base_values=expected_value,
            data=mol_explain,
            feature_names=feature_names,
        )

    shap.plots.waterfall(sample_exp, max_display=10, show=False)
    plt.title("SHAP Waterfall Plot")
    plt.tight_layout()
    plt.show()


def highlight_fragment(fragment_name, smiles, width=500, height=350):
    """
    Draw one fr_* fragment descriptor on a molecule, the descriptor-model
    equivalent of highlight_bit. Returns without drawing when the fragment is
    absent or its SMARTS is unavailable.
    """
    base = str(fragment_name).split(" ")[0]
    patt = _FRAGMENT_PATTERNS.get(base)
    if patt is None:
        print(f"{base} is not a fragment descriptor, nothing to draw. "
              f"Features like MolWt describe the whole molecule.")
        return

    mol = _mol_from_smiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string provided: {smiles}")
        return

    matches = mol.GetSubstructMatches(patt)
    if not matches:
        print(f"{base} not present in this molecule")
        return

    color = (0.1, 0.7, 0.3)
    atom_colors, bond_colors = {}, {}
    for match in matches:
        for atom_idx in match:
            atom_colors[atom_idx] = color
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                bond_colors[bond.GetIdx()] = color

    print(f"{base} is present {len(matches)}x in this molecule:")
    _draw_highlighted(mol, atom_colors, bond_colors, width, height)


def highlight_bit_on_molecule(
    bit_id, mol, width=500, height=350, X=None, smiles_list=None
):
    # ensure bit identifier is an integer index
    bit_id = int(bit_id)

    # handle smiles input
    smi = None
    if isinstance(mol, str):
        smi = mol
        mol = Chem.MolFromSmiles(smi)
    # non-string, non-Mol input (e.g. NaN from a CSV) is treated as invalid
    elif not isinstance(mol, Chem.Mol):
        mol = None

    if mol is None:
        print(f"Invalid SMILES string provided: {smi}")
        return

    bit_info = get_bit_info(mol)

    # check if the bit exists in this specific molecule
    if bit_id not in bit_info:
        print(f"Bit {bit_id} not present in molecule\n")

        # optionally search supplied dataset for a molecule containing the bit
        if X is not None and smiles_list is not None:
            highlight_bit(bit_id, X, smiles_list)
        return

    color = (0.1, 0.7, 0.3)
    atom_colors, bond_colors = {}, {}

    for central_atom_idx, radius in bit_info[bit_id]:
        atom_colors[central_atom_idx] = color
        if radius > 0:
            for bond_idx in Chem.FindAtomEnvironmentOfRadiusN(
                mol, radius, central_atom_idx
            ):
                bond_colors[bond_idx] = color
                bond = mol.GetBondWithIdx(bond_idx)
                atom_colors[bond.GetBeginAtomIdx()] = color
                atom_colors[bond.GetEndAtomIdx()] = color

    print(f"Bit {bit_id} is present in molecule:")
    _draw_highlighted(mol, atom_colors, bond_colors, width, height)


def highlight_bit(
    bit_id, X_matrix, smiles_list, smiles=None, shap_matrix=None, width=500, height=350
):
    # NOTE: shap_matrix, width and height are currently unused
    bit_id = int(bit_id)

    # support sparse matrices by converting to dense for bit lookup
    if hasattr(X_matrix, "toarray"):
        X_matrix = X_matrix.toarray()

    X_matrix = np.asarray(X_matrix)

    # ensure matrix is 2D so column indexing works for single-molecule inputs
    if X_matrix.ndim == 1:
        X_matrix = X_matrix.reshape(1, -1)

    smiles_list = np.asarray(smiles_list)

    # stop cleanly when requested bit index is outside fingerprint range
    if bit_id < 0 or bit_id >= X_matrix.shape[1]:
        print(f"Bit {bit_id} is outside the fingerprint range.")
        return

    # find indices of molecules containing the target bit
    present_indices = np.where(X_matrix[:, bit_id] > 0)[0]

    if len(present_indices) == 0:
        print(f"Bit {bit_id} is not present in any provided molecule.")
        return

    if smiles is None:
        sample_idx = present_indices[0]
        smi = smiles_list[sample_idx]
    else:
        sample_idx = None
        smi = smiles

    mol = _mol_from_smiles(smi)
    if mol is None:
        where = f" at index {sample_idx}" if sample_idx is not None else ""
        print(f"Invalid SMILES encountered{where}: {smi}")
        return

    bit_info = get_bit_info(mol)

    # check if the selected molecule actually contains the bit
    if bit_id not in bit_info:
        print(f"Bit {bit_id} is not present in the selected molecule.")
        return

    img = Draw.DrawMorganBit(mol, bit_id, bit_info)
    IPython.display.display(img)


# ============================================================
# EXAMPLES AND GLOBAL VIEW
# ============================================================
def select_examples(result, mask=None):
    """
    Pick test molecules that show what the model gets right and wrong.
    mask: optional boolean array over the test rows (e.g. in-domain only).
    Returns name -> (smiles, predicted probability, true label).
    Needs a smiles column in the test data.
    """
    test_df = result["test_df"].reset_index(drop=True)
    if "smiles" not in test_df.columns:
        print("No smiles column in the test data, cannot pick example molecules.")
        return {}

    proba = np.asarray(result["proba"])
    y_te = np.asarray(result["y_te"])

    # restrict to the allowed rows
    allowed = np.ones(len(y_te), dtype=bool) if mask is None else np.asarray(mask)
    active = np.where((y_te == 1) & allowed)[0]
    inactive = np.where((y_te == 0) & allowed)[0]

    # most confident hits and most confident mistakes (skip empty groups)
    picks = {}
    if len(active) > 0:
        picks["True active (most confident)"] = active[np.argmax(proba[active])]
        picks["False negative (active predicted inactive)"] = active[
            np.argmin(proba[active])
        ]
    if len(inactive) > 0:
        picks["True inactive (most confident)"] = inactive[np.argmin(proba[inactive])]
        picks["False positive (inactive predicted active)"] = inactive[
            np.argmax(proba[inactive])
        ]

    return {
        name: (test_df["smiles"].iloc[i], float(proba[i]), int(y_te[i]))
        for name, i in picks.items()
    }


def explain_molecule(
    smiles,
    model,
    target_class=1,
    feature_names=None,
    n_top=2,
    chembl_id=None,
    featurizer=None,
):
    """Explain one molecule: highlighted structure plus waterfall plot."""
    result = get_shap_features(
        smiles, model, target_class, chembl_id=chembl_id, featurizer=featurizer
    )
    if result is None:
        return

    mol_explain, id_explain, explainer, shap_class, sorted_features = result
    class_idx = _resolve_class_index(model, target_class)
    expected_value = get_class_expected_value(model, target_class)

    # model probability for the target class vs. the average prediction
    proba = model.predict_proba(mol_explain.reshape(1, -1))[0, class_idx]
    print(f"SMILES: {smiles}")
    print(f"Base value: {expected_value:.3f} | Model probability: {proba:.3f}")

    # molecule with the top contributing features highlighted
    visualize_mol_shap(
        smiles,
        model,
        target_class,
        n_top=n_top,
        chembl_id=chembl_id,
        featurizer=featurizer,
        feature_names=feature_names,
    )

    # waterfall plot of the strongest features
    waterfall_plot(shap_class, expected_value, mol_explain, feature_names)


def global_shap(model, X, target_class, n_max=500):
    """SHAP values for many molecules -> (shap matrix, matching feature matrix)."""
    X = np.asarray(X)
    if len(X) == 0:
        raise ValueError("No molecules to explain")

    # subsample for speed, TreeSHAP cost grows with the number of molecules
    if len(X) > n_max:
        rng = np.random.default_rng(RANDOM_STATE)
        X = X[rng.choice(len(X), size=n_max, replace=False)]

    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    # standardize SHAP output and select the target class
    shap_values = _standardize_shap_values(shap_values, model)
    class_idx = _resolve_class_index(model, target_class)
    shap_matrix = _select_class_shap(shap_values, model, class_idx)

    return shap_matrix, X


def plot_global_shap(shap_matrix, X, feature_names=None, max_display=20):
    """Beeswarm summary plot of the most important features."""
    shap.summary_plot(
        shap_matrix,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("Global SHAP Summary")
    plt.tight_layout()
    plt.show()


def top_features_table(shap_matrix, X, feature_names=None, n_top=15,
                       localisable_only=False):
    """
    Table of the most important features across molecules.

    localisable_only keeps only features that can be drawn on a structure:
    fingerprint bits for a fingerprint model, fr_* fragment counts for a
    descriptor model. Use it to pick something for highlight_bit or
    highlight_fragment.
    """
    X = np.asarray(X)
    n_features = shap_matrix.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    feature_names = list(feature_names)

    # mean absolute SHAP = overall importance of the feature
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    # share of molecules where the feature is non-zero
    present = X > 0
    share = present.mean(axis=0)

    # mean SHAP among molecules where it is present = direction of its effect
    n_present = np.maximum(present.sum(axis=0), 1)
    mean_present = (shap_matrix * present).sum(axis=0) / n_present

    kinds = np.array([feature_kind(n) for n in feature_names])

    ranking = np.argsort(mean_abs)[::-1]
    if localisable_only:
        ranking = np.array(
            [i for i in ranking if kinds[i] in ("fingerprint bit", "fragment count")]
        )
    order = ranking[:n_top]

    return pd.DataFrame(
        {
            "feature_index": order,
            "kind": kinds[order],
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": mean_abs[order],
            "mean_shap_when_present": mean_present[order],
            "pushes_toward": np.where(mean_present[order] > 0, "active", "inactive"),
            "share_of_molecules": share[order],
        }
    ).reset_index(drop=True)


def top_bits_table(shap_matrix, X, feature_names=None, n_top=15, fp_only=False):
    """Kept for existing notebooks; fp_only maps to localisable_only."""
    return top_features_table(
        shap_matrix, X, feature_names, n_top, localisable_only=fp_only
    )
