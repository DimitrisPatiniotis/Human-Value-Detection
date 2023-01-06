# Touché23-Human-Value-Detection
DOI: https://doi.org/10.5281/zenodo.6814563
Version: 2022-12-05

Dataset for [Touché / SemEval 2023 Task 4; ValueEval: Identification of Human Values behind Arguments](https://touche.webis.de/semeval23/touche23-web). Based on the original [Webis-ArgValues-22 dataset](https://doi.org/10.5281/zenodo.5657249) accompanying the paper [Identifying the Human Values behind Arguments](https://webis.de/publications.html#kiesel_2022b), published at ACL'22. The main dataset contains 8865 arguments.


## Argument Corpus
The annotated corpus in tab-separated value format. Contains the following files for the different dataset splits:
- `arguments-<split>.tsv`: Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - `Conclusion`: Conclusion text of the argument
    - `Stance`: Stance of the `Premise` towards the `Conclusion`; one of "in favor of", "against"
    - `Premise`: Premise text of the argument
- `labels-<split>.tsv`: Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - Other: Each other column corresponds to one value category, with a 1 meaning that the argument resorts to the value category and a 0 that not
- `level1-labels-<split>.tsv`: The same as `labels-<split>.tsv` but for the 54 level 1 values of the taxonomy (used in human annotation). Though not used for the 2023 task (except for the annotation), participants can still use them in their approaches.

For the main purpose of the ValueEval shared task, `<split>` is one of:
  - training               Arguments for training approaches (61%)
  - validation             Arguments for validating (optimizing) approaches (21%)
  - test                   Arguments for testing approaches; labels are not published (18%)

The distribution of argument sources is the same for training, validation, and test. Arguments with the same conclusion are always in the same split.

In addition, we provide the following datasets (by name for `<split>`) that contain different kinds of arguments. These are intended to test the robustness of approaches along with the main shared task evaluation:
  - validation-zhihu       Arguments from the recommendation and hotlist section of the Chinese question-answering website Zhihu, which teams can use for training or validating more robust approaches; these have been part of the original Webis-ArgValues-22 dataset

Added soon:
  - test-nahjalbalagha     Arguments from and based on the Nahj al-Balagha [1]; arguments contributed by the language.ml lab (Sina) [2]; labels are not published

[1] https://en.wikipedia.org/wiki/Nahj_al-Balagha
[2] https://language.ml


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
- Nailia Mirzakhmedova, Bauhaus-Universität Weimar, nailia.mirzakhmedova@uni-weimar.de 
- Milad Alshomary, Paderborn University, milad.alshomary@upb.de
- Maximilian Heinrich, Bauhaus-Universität Weimar, maximilian.heinrich@uni-weimar.de
- Nicolas Handke, Universität Leipzig, nh54fapa@studserv.uni-leipzig.de
- Xiaoni Cai, Technische Universität München, caix@in.tum.de
- Henning Wachsmuth, Paderborn University, henningw@upb.de
- Benno Stein, Bauhaus-Universität Weimar, benno.stein@uni-weimar.de


## Version History
- 2022-12-05
  - Complete data for the ValueEval shared task

- 2022-07-11
  - Exchanged the values.json from original dataset with task-specific value-categories.json

- 2022-07-09
  - Initial


## License
This dataset is distributed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

