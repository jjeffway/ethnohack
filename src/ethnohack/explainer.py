# load libraries
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from ethnohack.constants import MORGAN_RADIUS, FP_SIZE, RANDOM_STATE
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

# silence RDKit C++ warnings
RDLogger.DisableLog("rdApp.warning")

# cache explainers (building one for a large forest takes a while)
_explainer_cache = {}


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


# function to convert SMILES str -> fingerprint array
def smiles_to_ecfp4(smiles):
    # accept only string-like SMILES input
    if not isinstance(smiles, str):
        return None

    # use RDkit MolFromSmiles function to convert SMILES -> RDkit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # handle invalid SMILES -> return vector of NaN
    # changed to return None so callers can handle invalid input cleanly
    if mol is None:
        return None

    # extract Morgan Fingerprint (ECFP4) with feature information -> use RDkit fingerprint generator
    # NOTE: radius, size and bit (not count) type must match the fingerprints used to train the model
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=FP_SIZE
    )
    fp = generator.GetFingerprint(mol)

    # convert RDKit ExplicitBitVect -> NumPy array
    # use full fingerprint length so the NumPy buffer matches the RDKit vector
    # DataStructs is imported directly (AllChem.DataStructs only works by accident)
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


# resolve class label or class position to integer column position
def _resolve_class_index(model, target_class):
    # if model exposes classes_, map label to position first
    # labels take priority so labels like 1/2 are not mistaken for column positions
    if hasattr(model, "classes_"):
        classes = np.asarray(model.classes_)
        matches = np.where(classes == target_class)[0]
        if len(matches) > 0:
            return int(matches[0])

    # if target_class is already a valid integer position, keep it
    if isinstance(target_class, (int, np.integer)):
        return int(target_class)

    # fallback to integer conversion
    return int(target_class)


# standardize SHAP output so class axis is last when classes exist
def _standardize_shap_values(shap_values, model):
    # TreeExplainer returns arrays or lists depending on model and shap version
    if isinstance(shap_values, list):
        # list of class arrays -> stack to (n_samples, n_features, n_classes)
        return np.stack(shap_values, axis=-1)

    # convert to NumPy for uniform slicing
    shap_array = np.asarray(shap_values)

    # some shap versions return (n_classes, n_samples, n_features)
    if shap_array.ndim == 3:
        # infer class count when possible
        n_classes = len(getattr(model, "classes_", []))
        if n_classes == 0:
            n_classes = shap_array.shape[-1]

        # class axis already last -> keep (checked first so a batch of n_classes
        # molecules, e.g. 2 samples for a binary model, is not transposed by mistake)
        if shap_array.shape[-1] == n_classes:
            return shap_array

        # move class axis to last position when it is currently first
        if shap_array.shape[0] == n_classes:
            return np.transpose(shap_array, (1, 2, 0))

        return shap_array

    # binary/single-output models may return (n_samples, n_features)
    return shap_array


# get expected value for selected class from explainer
def _get_expected_value(explainer, class_idx, model=None):
    # read expected value from SHAP explainer
    expected = explainer.expected_value

    # convert array-like expected values to NumPy
    expected_array = np.asarray(expected)

    # scalar or single expected value is used for binary/single-output models
    if np.isscalar(expected) or expected_array.size == 1:
        value = float(expected_array.ravel()[0])

        # single-output binary models (e.g. XGBoost, LightGBM) explain the positive class only
        # flip the sign when the negative class (position 0) is requested
        if (
            model is not None
            and len(getattr(model, "classes_", [])) == 2
            and class_idx == 0
        ):
            value = -value

        return value

    # otherwise select class expected value
    return float(expected_array[class_idx])


# select one class from standardized SHAP output -> (n_samples, n_features)
def _select_class_shap(shap_values, model, class_idx):
    # slice class column (class shapley values)
    if shap_values.ndim == 3:
        return shap_values[:, :, class_idx]

    # fallback for binary/single-output SHAP arrays
    shap_class = np.asarray(shap_values)

    # single-output binary models explain the positive class only
    # flip the sign when the negative class (position 0) is requested
    if len(getattr(model, "classes_", [])) == 2 and class_idx == 0:
        shap_class = -shap_class

    return shap_class


# get expected value for a target class without manually inspecting explainer
def get_class_expected_value(model, target_class):
    # reuse cached explainer to read expected value
    explainer = _get_explainer(model)

    # resolve target class label to integer position
    class_idx = _resolve_class_index(model, target_class)

    return _get_expected_value(explainer, class_idx, model)


# calculate SHAP values -> return sorted bits (and other relevant variables)
def get_shap_bits(smiles, model, target_class, chembl_id=None):
    # get fingerprint for molecule
    mol_explain = smiles_to_ecfp4(smiles)

    # stop cleanly when SMILES cannot be parsed
    if mol_explain is None:
        print(f"Invalid SMILES string: {smiles}")
        return None

    # get Molecule ChEMBL ID for molecule
    # pass ID directly because dataset/model are defined outside this module
    id_explain = str(chembl_id) if chembl_id is not None else None

    # set up the explainer using shap TreeExplainer for the model (cached)
    explainer = _get_explainer(model)

    # get the shapley values for each bit in molecule fingerprint
    # reshape to (1, n_features) for SHAP prediction
    shap_values = explainer.shap_values(
        mol_explain.reshape(1, -1), check_additivity=False
    )

    # standardize SHAP output across SHAP versions and model types
    shap_values = _standardize_shap_values(shap_values, model)

    # resolve target class label to integer position
    class_idx = _resolve_class_index(model, target_class)

    # slice class column (class shapley values) for the single molecule
    shap_class = _select_class_shap(shap_values, model, class_idx)[0]

    # map bit index to shapley value
    values_shap = {bit_id: shap_val for bit_id, shap_val in enumerate(shap_class)}

    # sort bits by absolute SHAP value
    sorted_bits = sorted(values_shap.items(), key=lambda x: abs(x[1]), reverse=True)

    return mol_explain, id_explain, explainer, shap_class, sorted_bits


# show molecule with top contributing bits highlighted
def visualize_mol_shap(
    smiles,
    model,
    target_class,
    n_top=2,
    chembl_id=None,
):
    # convert to RDKit mol -> generate bitInfo Map
    # helper handles NaN/non-string input without raising
    mol = _mol_from_smiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return

    # generate bit information for the molecule
    bit_info = get_bit_info(mol)

    # get sorted SHAP bits (tuple list: [(bit_id, shap_val), ...])
    # pass ChEMBL ID directly instead of using a dataframe inside explainer.py
    result = get_shap_bits(
        smiles,
        model,
        target_class,
        chembl_id=chembl_id,
    )

    # stop cleanly when fingerprint or SHAP calculation fails
    if result is None:
        return

    mol_explain, id_explain, explainer, shap_class, sorted_bits = result

    # show molecule identifier when supplied
    if id_explain is not None:
        print(f"Molecule ChEMBL ID: {id_explain}")

    # filter strictly for present bits (bit_id in bit_info)
    present = [b for b in sorted_bits if b[0] in bit_info]

    # select top N positive and top N negative bits
    positive_bits = [(b, v) for b, v in present if v > 0][:n_top]
    negative_bits = [(b, v) for b, v in present if v < 0][:n_top]
    selected = positive_bits + negative_bits

    # sort selected bits by absolute SHAP value for stable display order
    selected = sorted(selected, key=lambda x: abs(x[1]), reverse=True)

    # if no signed bits are available, fall back to strongest present bits
    # (these have SHAP value 0, so they are drawn in neutral gray below)
    if not selected:
        selected = present[:n_top]

    if not selected:
        print("No active present bits found to highlight.")
        return

    # prepare dictionaries for single-plot overlay
    atom_colors = {}
    bond_colors = {}
    atoms_to_highlight = set()
    bonds_to_highlight = set()

    # define RGB color palette: red for active, blue for inactive, gray for zero contribution
    RED = (1.0, 0.2, 0.2)
    BLUE = (0.2, 0.4, 1.0)
    GRAY = (0.6, 0.6, 0.6)

    print(f"=== Overlaying Top {len(selected)} Present Features ===")
    for bit_id, shap_val in selected:
        if shap_val > 0:
            direction = "Active (+)"
        elif shap_val < 0:
            direction = "Inactive (-)"
        else:
            direction = "Neutral (0)"
        print(f"Bit ID: {bit_id:4d}| SHAP: {shap_val:+.6f}| {direction}")

    # color weakest bits first so the strongest bits win where environments overlap
    # (later assignments overwrite earlier ones in the color dictionaries)
    for bit_id, shap_val in reversed(selected):
        color = RED if shap_val > 0 else BLUE if shap_val < 0 else GRAY

        # extract central atom and radius for each occurrence of the bit
        for central_atom_idx, radius in bit_info[bit_id]:
            atoms_to_highlight.add(central_atom_idx)
            atom_colors[central_atom_idx] = color

            if radius > 0:
                env_bonds = Chem.FindAtomEnvironmentOfRadiusN(
                    mol, radius, central_atom_idx
                )
                for bond_idx in env_bonds:
                    bonds_to_highlight.add(bond_idx)
                    bond_colors[bond_idx] = color
                    bond = mol.GetBondWithIdx(bond_idx)
                    u = bond.GetBeginAtomIdx()
                    v = bond.GetEndAtomIdx()
                    atoms_to_highlight.update([u, v])
                    atom_colors[u] = color
                    atom_colors[v] = color

    # render single consolidated drawing
    drawer = rdMolDraw2D.MolDraw2DSVG(790, 600)
    options = drawer.drawOptions()

    # keeps non-highlighted atoms grayscale/clean
    options.useBWAtomPalette()
    options.continuousHighlight = True

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atoms_to_highlight),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bonds_to_highlight),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    IPython.display.display(IPython.display.SVG(drawer.GetDrawingText()))


# create waterfall plot for contributing bits
def waterfall_plot(shap_class, expected_value, mol_explain, feature_names=None):
    # build explanation object
    # include optional feature names when supplied by caller
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

    # waterfall plot
    shap.plots.waterfall(sample_exp, max_display=10, show=False)
    plt.title("SHAP Waterfall Plot")
    plt.tight_layout()
    plt.show()


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

    # generate Morgan Fingerprint with BitInfo Map for this molecule
    bit_info = get_bit_info(mol)

    # check if the bit exists in this specific molecule
    if bit_id not in bit_info:
        print(f"Bit {bit_id} not present in molecule")
        print()

        # optionally search supplied dataset for a molecule containing the bit
        if X is not None and smiles_list is not None:
            highlight_bit(bit_id, X, smiles_list)
        return

    # set highlight color to green
    color = (0.1, 0.7, 0.3)

    # extract central atoms and radius environments
    atoms_to_highlight = set()
    bonds_to_highlight = set()
    atom_colors = {}
    bond_colors = {}

    for central_atom_idx, radius in bit_info[bit_id]:
        atoms_to_highlight.add(central_atom_idx)
        atom_colors[central_atom_idx] = color

        if radius > 0:
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, central_atom_idx)
            for bond_idx in env_bonds:
                bonds_to_highlight.add(bond_idx)
                bond_colors[bond_idx] = color

                # add attached atoms at the ends of environment bonds
                bond = mol.GetBondWithIdx(bond_idx)
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                atoms_to_highlight.update([u, v])
                atom_colors[u] = color
                atom_colors[v] = color

    # render SVG drawing
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()  # keep non-highlighted atoms in clean grayscale
    options.continuousHighlight = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atoms_to_highlight),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bonds_to_highlight),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    # display SVG in notebook
    print(f"Bit {bit_id} is present in molecule:")
    print(f"Displaying Bit ID {bit_id}")
    IPython.display.display(IPython.display.SVG(drawer.GetDrawingText()))


def highlight_bit(
    bit_id, X_matrix, smiles_list, smiles=None, shap_matrix=None, width=500, height=350
):
    # NOTE: shap_matrix, width and height are currently unused
    # ensure bit identifier is an integer index
    bit_id = int(bit_id)

    # support sparse matrices by converting to dense for bit lookup
    if hasattr(X_matrix, "toarray"):
        X_matrix = X_matrix.toarray()

    # convert to NumPy so boolean/integer indexing behaves consistently
    X_matrix = np.asarray(X_matrix)

    # ensure matrix is 2D so column indexing works for single-molecule inputs
    if X_matrix.ndim == 1:
        X_matrix = X_matrix.reshape(1, -1)

    # convert SMILES container to NumPy for consistent positional access
    smiles_list = np.asarray(smiles_list)

    # stop cleanly when requested bit index is outside fingerprint range
    if bit_id < 0 or bit_id >= X_matrix.shape[1]:
        print(f"Bit {bit_id} is outside the fingerprint range.")
        return

    # find indices of molecules containing the target bit
    present_indices = np.where(X_matrix[:, bit_id] == 1)[0]

    if len(present_indices) == 0:
        print(f"Bit {bit_id} is not present in any provided molecule.")
        return

    if smiles is None:
        # pick the first representative molecule
        sample_idx = present_indices[0]
        smi = smiles_list[sample_idx]
    else:
        # use caller-supplied SMILES and track that no dataset index was selected
        sample_idx = None
        smi = smiles

    # helper handles NaN/non-string entries without raising
    mol = _mol_from_smiles(smi)
    if mol is None:
        if sample_idx is not None:
            print(f"Invalid SMILES encountered at index {sample_idx}: {smi}")
        else:
            print(f"Invalid SMILES provided: {smi}")
        return

    # generate Morgan Fingerprint with BitInfo Map
    bit_info = get_bit_info(mol)

    # check if the selected molecule actually contains the bit
    if bit_id not in bit_info:
        print(f"Bit {bit_id} is not present in the selected molecule.")
        return

    # draw the requested Morgan bit on the molecule
    img = Draw.DrawMorganBit(mol, bit_id, bit_info)
    IPython.display.display(img)


# pick test molecules that show what the model gets right and wrong
def select_examples(result):
    # reset index so positions match the probability/label arrays
    test_df = result["test_df"].reset_index(drop=True)
    proba = np.asarray(result["proba"])
    y_te = np.asarray(result["y_te"])
    active = np.where(y_te == 1)[0]
    inactive = np.where(y_te == 0)[0]

    # most confident hits and most confident mistakes
    picks = {
        "True active (most confident)": active[np.argmax(proba[active])],
        "True inactive (most confident)": inactive[np.argmin(proba[inactive])],
        "False positive (inactive predicted active)": inactive[
            np.argmax(proba[inactive])
        ],
        "False negative (active predicted inactive)": active[np.argmin(proba[active])],
    }

    # return name -> (smiles, predicted probability, true label)
    return {
        name: (test_df["smiles"].iloc[i], float(proba[i]), int(y_te[i]))
        for name, i in picks.items()
    }


# explain one molecule: highlighted structure + waterfall plot
def explain_molecule(
    smiles, model, target_class=1, feature_names=None, n_top=2, chembl_id=None
):
    # get fingerprint, SHAP values and expected value for the molecule
    result = get_shap_bits(smiles, model, target_class, chembl_id=chembl_id)
    if result is None:
        return

    mol_explain, id_explain, explainer, shap_class, sorted_bits = result
    class_idx = _resolve_class_index(model, target_class)
    expected_value = get_class_expected_value(model, target_class)

    # model probability for the target class vs. the average prediction (base value)
    proba = model.predict_proba(mol_explain.reshape(1, -1))[0, class_idx]
    print(f"SMILES: {smiles}")
    print(f"Base value: {expected_value:.3f} | Model probability: {proba:.3f}")

    # molecule with top contributing bits highlighted
    visualize_mol_shap(smiles, model, target_class, n_top=n_top, chembl_id=chembl_id)

    # waterfall plot of the strongest bits
    waterfall_plot(shap_class, expected_value, mol_explain, feature_names)


# calculate SHAP values for many molecules -> (shap matrix, matching feature matrix)
def global_shap(model, X, target_class, n_max=500):
    X = np.asarray(X)

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


# beeswarm summary plot of the most important bits
def plot_global_shap(shap_matrix, X, feature_names=None, max_display=20):
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


# table of the most important bits across molecules
def top_bits_table(shap_matrix, X, feature_names=None, n_top=15):
    X = np.asarray(X)
    n_features = shap_matrix.shape[1]
    if feature_names is None:
        feature_names = [f"bit{i}" for i in range(n_features)]

    # mean absolute SHAP = overall importance of the bit
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    # share of molecules containing the bit
    present = X > 0
    share = present.mean(axis=0)

    # mean SHAP among molecules that contain the bit = direction of its effect
    n_present = np.maximum(present.sum(axis=0), 1)
    mean_present = (shap_matrix * present).sum(axis=0) / n_present

    order = np.argsort(mean_abs)[::-1][:n_top]
    table = pd.DataFrame(
        {
            "bit": order,
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": mean_abs[order],
            "mean_shap_when_present": mean_present[order],
            "pushes_toward": np.where(mean_present[order] > 0, "active", "inactive"),
            "share_of_molecules": share[order],
        }
    )
    return table.reset_index(drop=True)
