import csv

def main():
    with open('data/misc/eval_metrics.csv', mode ='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            print(lines)

if __name__ == '__main__':
    main()
