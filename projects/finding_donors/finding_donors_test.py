import pandas as pd
import matplotlib.pyplot as plt


def get_data(file_path):
    data = pd.read_csv(file_path)

    # print('\n====================================================\n')
    # print(data.head(30))
    # print('\n====================================================\n')

    return data


def process(data):
    data['income'] = data['income'].apply(lambda income: 0 if income == '<=50K' else 1)
    return data


def draw_histogram(values, feature_name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, color='#00A0A0', bins=20)

    ax.set_title("Histogram for feature: '{}'".format(feature_name), fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Number of Records")
    # ax.set_ylim((0, 2000))
    # ax.set_yticks([0, 500, 1000, 1500, 2000])
    # ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    fig.tight_layout()
    fig.show()

    return


def corr(dataframe, categorical=False):
    return


def main(census_file):
    census_data = get_data(census_file)
    census_data = process(census_data)

    print(census_data.head())
    print('\n\n===========\n\n')

    # print(census_data.head())
    #
    # fs =['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    #
    # for feature_name in fs:
    #     draw_histogram(census_data[feature_name].tolist(), feature_name)

    # numeric_fields = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # non_numeric_fields = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    #
    # # Correlations with numeric values
    # for numeric_field in numeric_fields:
    #     corr = census_data[[numeric_field, 'income']].corr()[numeric_field]['income']
    #     print('Correlation between {} and income: {}'.format(numeric_field, corr))
    # print('\n\n===========================\n\n')
    #
    # # Groupby for categorical variables
    # for non_numeric_field in non_numeric_fields:
    #     rel_data = census_data[[non_numeric_field, 'income']]
    #     group_data = rel_data.groupby(non_numeric_field).mean().reset_index()
    #     group_data.sort_values('income', inplace=True, ascending=True)
    #
    #     print('For different groups of feature "{}", proportion of people with more than 50k salary:'.format(non_numeric_field))
    #     print(group_data)
    #     print('\n')
    #
    #
    #
    # print('\n\n===========================\n\n')


    return


if __name__ == '__main__':
    census_file = '/Users/sravan/Desktop/projects/machine-learning/projects/finding_donors/census.csv'
    main(census_file)