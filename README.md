# ethnohack
Repository for EthnoHack 2026 Data Hackathon on Ethnopharmacology &amp; Health submission

## Gameplan
1. Choose dataset (active/inactive compounds) for two complementary targets (A and B) that participate in synergistic pathways
2. Retrieve dataset, filter out unsuitable value
3. Train tree-ensemble model (Random Forest or XGBoost) on each dataset (including hyperparameter optimization, scaffold split, Optuna?)
4. Evaluate each model separately (ROC-AUC, balanced accuracy, PR-AUC with active-prevalence baseline?, MCC?, y-scrambling control?)
5. Run TreeSHAP explainer to identify important chemical features for activity
6. Score COCONUT/NPASS compounds with the model directly, identify important features and cross-check explanations (max Tanimoto applicability domain?)
7. Validate by checking a relevant plant with a known synergistic pair matching the targets, also check negative control plants
8. Run through ADMET filters, Lipinski Rule of Five
9. Identify plants that have two synergistic compounds (pA(c1) × pB(c2) for c1 ≠ c2 in the same plant)
10. Flag compounds that score high on both models as dual-target, and report them separately from pairs
11. If lacking viable results, also check inter-plant compound pairs
12. Output a complementary-mechanism hypothesis, identify novel synergistic pairs of natural compounds found in the same plant


## Potential Targets
- COX-2 + 5-LOX (inflammation)
- COX-2 + AChE (neuroinflammation)

COX-2 chEMBL: https://www.ebi.ac.uk/chembl/explore/target/CHEMBL230


## Christian's Hints
- Find plants that have natural compounds that can be used as a drug, or known drugs that have other applications
- chebi -> chemicals of biolohical interest
- opentargets -> in clinical trial or approved by FDA
- taxtree -> NCBI taxonomy tree
- icd11 -> categorize diseases
- ipni -> classify plants (more specialized than NCBI)
- wcvp -> location of plants
- wfo -> for uncommon nicknames of plants
- obo -> open biomedical ontologies


***


# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.