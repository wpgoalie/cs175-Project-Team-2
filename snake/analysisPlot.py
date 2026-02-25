import csv
import pandas as pd



def main():
    # with open('data/misc/eval_metrics.csv', mode ='r') as file:
    #     csvFile = csv.reader(file)
    #     for lines in csvFile:
    #         print(f'{lines[0]} | {lines[1]} | ')

    # Read CSV
    df = pd.read_csv("data/misc/eval_metrics.csv")
    
    # Convert to Markdown table
    print(df.to_markdown(index=False))

if __name__ == '__main__':
    main()
