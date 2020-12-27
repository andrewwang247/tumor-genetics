"""
Analyze COVID-19 data for statistical patterns.

Copyright 2020. Siwei Wang.
"""
from os import path
from csv import writer
from typing import List
from itertools import combinations, product
from math import isnan
from timeit import default_timer as timer
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pandas import DataFrame, read_csv  # type: ignore
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import pearsonr  # type: ignore


def data_frame() -> DataFrame:
    """Retrieve the data frame from file."""
    data_file = 'data.csv'
    assert path.splitext(data_file)[1] == '.csv'
    frame = read_csv(data_file)
    print(f'Imported frame with shape {frame.shape} from {data_file}')
    return frame


def percent_difference(frame: DataFrame,
                       cancers: List[str],
                       genes: List[str]) -> np.ndarray:
    """Retrieve percent differences from data frame."""
    percent_diff = np.empty(shape=(len(cancers), len(genes)), dtype=float)
    for idx, gene in enumerate(genes):
        normal = frame[f'{gene} Normal']
        tumor = frame[f'{gene} Tumor']
        percent_diff[:, idx] = [0.0 if isnan(num)
                                else num for num in
                                (tumor - normal) / normal]
    return percent_diff


def write_percents_csv(
        fname: str,
        data: np.ndarray,
        genes: List[str],
        cancers: List[str]):
    """Write percent difference matrix to a csv file."""
    with open(fname, 'w') as fhand:
        cwr = writer(fhand, delimiter=',', quotechar='"')
        cwr.writerow(['Cancer'] + [str(gene) for gene in genes])
        for idx, cancer in enumerate(cancers):
            cwr.writerow([cancer] + [str(round(pdiff, 3))
                                     for pdiff in data[idx, :]])


def write_gene_csv(fname: str, data: np.ndarray, genes: List[str]):
    """Write gene matrix to a csv file."""
    assert path.splitext(fname)[1] == '.csv'
    with open(fname, 'w') as fhand:
        cwr = writer(fhand, delimiter=',', quotechar='"')
        cwr.writerow(['Gene'] + [str(gene) for gene in genes])
        for idx, gene in enumerate(genes):
            cwr.writerow([gene] + [str(round(datum, 3))
                                   for datum in data[idx, :]])


def scatter_plot(name_i: str, name_j: str,
                 gene_i: List[float], gene_j: List[float]):
    """Draw a scatter plot for the two genes from their data."""
    plt.figure()
    assert len(gene_i) == len(gene_j)
    num_cancers = len(gene_i)
    plt.scatter(gene_i, gene_j, c='#fa7c69')
    poly = np.polyfit(gene_i, gene_j, 1)
    reg = np.poly1d(poly)
    equation = f'y = {round(poly[0], 3)} x ' + \
        (f'+ {round(poly[1], 3)}' if poly[1]
         >= 0 else f'- {abs(round(poly[1], 3))}')
    plt.plot(gene_i, reg(gene_i), label=equation, color='#721b75')
    plt.xlabel(f'{name_i} Percent Difference')
    plt.ylabel(f'{name_j} Percent Difference')
    plt.title(f'{name_i} And {name_j} Over {num_cancers} Cancers')
    plt.legend(loc='best')
    plt.savefig(f'{name_i}_{name_j}.png')
    plt.close()


def main():
    """Analyze COVID-19 data for statistical patterns."""
    begin = timer()
    frame = data_frame()
    cancers = list(frame['Cancer'])
    num_cancers = len(cancers)
    print(f'Detected {num_cancers} cancers: ' + ' '.join(cancers))
    genes = [col.split()[0] for col in frame.columns[1::2]]
    num_genes = len(genes)
    print(f'Detected {num_genes} genes: ' + ' '.join(genes))
    print('Calculating percent differences between cancers...')
    percent_diff = percent_difference(frame, cancers, genes)
    write_percents_csv('percent_diff.csv', percent_diff, genes, cancers)
    print('Computing correlations and matches between genes...')
    correlation = np.empty(shape=(num_genes, num_genes), dtype=float)
    matching = np.empty(shape=(num_genes, num_genes), dtype=int)
    for i, j in product(range(num_genes), repeat=2):
        gene_i = percent_diff[:, i]
        gene_j = percent_diff[:, j]
        correlation[i, j] = pearsonr(gene_i, gene_j)[0]
        matching[i, j] = np.count_nonzero(gene_i * gene_j >= 0)
    write_gene_csv('pearson_correlation.csv', correlation, genes)
    write_gene_csv('matching.csv', matching, genes)
    print('Generating scatter plots...')
    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        for i, j in combinations(range(num_genes), 2):
            pool.submit(scatter_plot,
                        genes[i],
                        genes[j],
                        percent_diff[:, i],
                        percent_diff[:, j])
    end = timer()
    print(f'Execution finished in {round(end - begin, 2)} seconds.')


if __name__ == '__main__':
    main()
