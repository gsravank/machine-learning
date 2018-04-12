import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_data():
    data = pd.read_csv('/Users/sravan/Desktop/projects/machine-learning/projects/customer_segments/customers.csv')
    return data


def pair_wise_scatter_plots(data, columns):
    num_columns = len(columns)
    num_plots = (num_columns * (num_columns - 1)) / 2

    fig = plt.figure(figsize=(20,60))

    i = 1
    for idx1 in range(num_columns):
        for idx2 in range(idx1 + 1, num_columns, 1):
            col1 = columns[idx1]
            col2 = columns[idx2]

            ax = fig.add_subplot(num_plots, 1, i)
            i += 1

            ax.scatter(data[col1], data[col2])
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)

    fig.tight_layout()
    fig.show()


def triple_wise_scatter_plots(data, columns):
    num_columns = len(columns)
    num_plots = (num_columns * (num_columns - 1)) / 2

    fig = plt.figure(figsize=(20,60))

    i = 1
    for idx1 in range(num_columns):
        for idx2 in range(idx1 + 1, num_columns, 1):
            for idx3 in range(idx2 + 1, num_columns, 1):
                col1 = columns[idx1]
                col2 = columns[idx2]
                col3 = columns[idx3]

                ax = fig.add_subplot(num_plots, 1, i, projection='3d')
                i += 1

                ax.scatter(data[col1], data[col2], data[col3])
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_zlabel(col3)

    fig.tight_layout()
    fig.show()


def main():
    customer_data = get_data()
    rel_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Delicatessen']
    customer_data = customer_data[rel_columns]

    # plt.hist(customer_data['Delicatessen'], bins=100)
    # plt.show()

    print(len(customer_data[customer_data['Delicatessen'] > 10000]))

    # Print pair wise scatter plots to
    # pair_wise_scatter_plots(customer_data, rel_columns)

    # triple_wise_scatter_plots(customer_data, rel_columns)

    # print(customer_data[(customer_data['Fresh'] >= 12000.0)
    #                     & (customer_data['Milk'] >= 5796.0)
    #                     & (customer_data['Grocery'] >= 7951.0)
    #                     & (customer_data['Frozen'] >= np.mean(customer_data['Frozen']))
    #       ]
    #       )

    print(np.mean(customer_data['Frozen']))

    return


if __name__ == '__main__':
    main()