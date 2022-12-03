# Touché23-Human-Value-Detection
DOI: https://doi.org/10.5281/zenodo.6814563
Version: 2022-07-11

Dataset for [Touché / SemEval 2023 Task 4; ValueEval: Identification of Human Values behind Arguments](https://touche.webis.de/semeval23/touche23-web). Based on the original [Webis-ArgValues-22 dataset](https://doi.org/10.5281/zenodo.5657249) accompanying the paper [Identifying the Human Values behind Arguments](https://webis.de/publications.html#kiesel_2022b), published at ACL'22.

The dataset currently contains 5220 arguments. We are, however, looking for more argument datasets (conclusion + stance + premise) to annotate and incorporate, especially datasets from different cultures and genres. Please send suggestions to our [task](mailto:valueeval@googlegroups.com) or [organizers](mailto:valueeval-organizers@googlegroups.com) mailing lists.


## Argument Corpus
The annotated corpus in tab-separated value format. Future versions of this dataset will contain more arguments and be split into "-training", "-validation", and "-testing" files to represent the corresponding sets for the evaluation.
- `arguments-training.tsv`: Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - `Conclusion`: Conclusion text of the argument
    - `Stance`: Stance of the `Premise` towards the `Conclusion`; one of "in favor of", "against"
    - `Premise`: Premise text of the argument
- `labels-training.tsv`: Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - Other: Each other column corresponds to one value category, with a 1 meaning that the argument resorts to the value category and a 0 that not
- `level1-labels-training.tsv`: The same as `labels-training.tsv` but for the 54 level 1 values of the taxonomy (used in human annotation). Though not used for the 2023 task (except for the annotation), participants can still use them in their approaches.


## Value Taxonomy
The `value-categories.json` describes the 20 value categories of this task through examples. Format:
```
{
  "<value category>": {
    "<level 1 value>": [
      "<exemplary effect a corresponding argument might target>",
      ...
    ], ...
  }, ...
}
```
The level 1 values are not used for the 2023 task (except for the annotation), but are still listed here for some might find them useful for understanding the value categories. See our paper on [Identifying the Human Values behind Arguments](https://webis.de/publications.html#kiesel_2022b) for the complete taxonomy.


## Authors
- Johannes Kiesel, Bauhaus-Universität Weimar, johannes.kiesel@uni-weimar.de
- Milad Alshomary, Paderborn University, milad.alshomary@upb.de
- Nicolas Handke, Universität Leipzig, nh54fapa@studserv.uni-leipzig.de
- Xiaoni Cai, Technische Universität München, caix@in.tum.de
- Henning Wachsmuth, Paderborn University, henningw@upb.de
- Benno Stein, Bauhaus-Universität Weimar, benno.stein@uni-weimar.de


## Version History
- 2022-07-11
  - Exchanged the values.json from original dataset with task-specific value-categories.json

- 2022-07-09
  - Initial


## License
This dataset is distributed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

