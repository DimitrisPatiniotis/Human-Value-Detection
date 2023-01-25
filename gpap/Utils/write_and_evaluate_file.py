import argparse
import csv
import os

def writeRun(labels, argument_ids, outputDataset):

    """
    :param labels: Pandas dataframe with 20 columns in the same order as in 'values' list, and N rows according to the dataset.
                   Each cell contains 1 or 0.
    :param argument_ids: List with N argument ids. The index of the list corresponds to the row index of 'labels'.
    :param outputDataset: The path for the output .tsv file.
    :return:
    """

    values = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
              "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
              "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
              "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance",
              "Universalism: objectivity"]

    assert labels.columns.tolist() == values, "Values used do not match the 'value' list !!!"

    #Tranform labels to the appropriate form
    temp_labels = {}
    for idx, argument_id in enumerate(argument_ids):
        temp_labels[argument_id] = [col for col in labels.columns.tolist() if int(labels[col].iloc[idx]) == 1]

    labels = temp_labels

    if not os.path.exists(outputDataset):
        os.makedirs(outputDataset)

    usedValues = set()
    for instanceValues in labels.values():
        usedValues.update(instanceValues)

    for usedValue in usedValues:
        if usedValue not in values:
            print("Unknown value: '" + usedValue + "'")
            exit(1)

    print("Detected values: " + str(usedValues))

    fieldNames = [ "Argument ID" ]
    for value in values:
        fieldNames.append(value)

    print("Writing run file")
    with open(os.path.join(outputDataset, "run.tsv"), "w") as runFile:
        writer = csv.DictWriter(runFile, fieldnames = fieldNames, delimiter = "\t")
        writer.writeheader()
        for (argumentId, instanceValues) in labels.items():
            row = { "Argument ID": argumentId }
            for value in values:
                if value in instanceValues:
                    row[value] = "1"
                else:
                    row[value] = "0"
            writer.writerow(row)

def evaluateRun(input_data_path, input_run_path, output_path):
    os.system('python ./../Utils/evaluator.py --inputDataset ' + input_data_path +
              ' --inputRun ' + input_run_path +
              ' --outputDataset ' + output_path)
