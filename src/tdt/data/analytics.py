from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import polars as pl
import polars.selectors as cs
from typing import Union, List, AnyStr
import numpy as np
import logging

log = logging.getLogger(__name__)


def transform_pca(x: Union[pl.DataFrame, pd.DataFrame], n_components=3):
    """
    Performs Principal Component Analysis (PCA) and data transformation on the input
    DataFrame and returns the PCA-transformed data, reconstructed data, and PCA model.

    The function accepts either a pandas DataFrame or a polars DataFrame as input. It
    applies scaling and PCA transformation using a pipeline, and reconstructs the data
    from the transformed components. The function allows customization of the number
    of principal components.

    Args:
        x: A pandas or polars DataFrame containing the data to be transformed. The first
            column is assumed to be an index column, and the remaining columns are treated
            as features for PCA.
        n_components: The number of principal components to compute during PCA. Defaults
            to 3.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, PCA]:
            - A polars DataFrame containing the PCA-transformed components with indices
              and standardized component names (either 'level', 'slope', 'curve' for 3
              components or dynamically named as 'PC_1', 'PC_2', etc.).
            - A polars DataFrame containing the reconstructed data created by reversing
              the PCA transformation.
            - The PCA model used for the transformation, allowing access to additional
              information such as explained variance ratio.
    """
    if isinstance(x, pl.DataFrame):
        x = x.to_pandas()
    cols = x.columns[1:]
    idx = x.iloc[:, 0]
    data = x[cols]
    pca = PCA(n_components=n_components)
    scaler = StandardScaler()
    pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
    c = ['level', 'slope', 'curve'] if n_components == 3 else [f'PC_{x}' for x in range(1, n_components + 1)]
    pcs = pd.DataFrame(data=pipeline.fit_transform(data), index=idx, columns=c)
    reconstructed = pl.DataFrame(
        pd.DataFrame(data=pipeline.inverse_transform(pcs), index=idx, columns=cols).reset_index())
    return pl.DataFrame(pcs.reset_index()).with_columns(pl.col('dt').cast(pl.Date)), reconstructed, pca


def solve_for_neutrality(pc_loadings, long_col, exposure_to):
    """
    Solves for PCA exposure neutrality in a given dataset based on principal component loadings.

    This function calculates weights to ensure neutrality with respect to a
    specified factor in the dataset. The solution involves solving a system
    of linear equations using the provided principal component loadings.

    Args:
        pc_loadings: DataFrame containing the principal component loadings for
            various factors and observations.
        long_col: str. The name of the column in `pc_loadings` that corresponds
            to the dependent variable for the linear system.
        exposure_to: Any. The index in `pc_loadings` that should be excluded
            from the calculations for the neutrality solution.

    Returns:
        list: A list of weights that solve for neutrality, where the neutrality
        criterion is satisfied for the specified dataset and factor.
    """
    rows = pc_loadings[pc_loadings.index != exposure_to]
    rhs = rows[long_col]
    lhs = rows.drop(columns=[long_col])
    wts = np.linalg.solve(lhs, rhs)
    wts = [wts[0], 1.00, wts[1]]
    return wts


def compute_weighted_butterfly_levels(levels: pl.DataFrame, durations: pl.DataFrame, wings: List[AnyStr], belly: AnyStr):
    """
    Computes weighted butterfly levels based on market value, DV01, and PCA neutrality adjustments
    for specified wings and belly maturities.

    This function calculates market value weighted butterfly levels using given durations and maturities,
    neutralizes principal component analysis (PCA) movements through adjustments, and aggregates these adjustments
    into the provided levels dataframe. The resulting adjustments aim to account for level, slope, and curvature
    exposures while focusing on maintaining neutrality across the specified wings and belly.

    Args:
        levels (pl.DataFrame): A dataframe that contains the yield curve levels for different maturities.
        durations (pl.DataFrame): A dataframe representing the duration exposures by maturity.
        wings (List[AnyStr]): A list containing the two wing maturities for the butterfly calculation.
        belly (AnyStr): The belly maturity for the butterfly calculation.

    Returns:
        pl.DataFrame: A dataframe containing the computed levels for market value, DV01, and PCA neutralized butterfly rates.
    """
    # Calculate DV01 weighted BF
    wts = durations \
        .select(['dt'] + wings + [belly]) \
        .select(pl.col(belly).truediv(pl.col(wings[0])).truediv(2).alias(f'wt_{wings[0]}'),
                pl.col(belly).truediv(pl.col(belly)).alias(f'wt_{belly}'),
                pl.col(belly).truediv(pl.col(wings[1])).truediv(2).alias(f'wt_{wings[1]}')).row(0)

    # Calculate PCA on changes to neutralize 1d PCA move
    pca_components, reconstructed, pca_diff = transform_pca(levels.with_columns(cs.numeric().diff()).drop_nulls(),
                                                            n_components=3)
    exposures = pd.DataFrame(pca_diff.components_, index=['level', 'slope', 'curve'])\
        .rename(columns={k: v + 'y' for k, v in enumerate(levels.select(cs.numeric()).columns)})
    e = exposures[[f'{m}y' for m in [wings[0], belly, wings[1]]]]
    pca_wts = solve_for_neutrality(e, belly+'y', 'curve')

    e2 = exposures.apply(lambda x: x/x.sum(),axis='columns')[e.columns]
    pca_wts2= list(e2.loc['curve',:].values.ravel()/e2.loc['curve',belly+'y'])
    print(pca_wts2)

    # combine into levels for yield
    rates = levels \
        .with_columns(
        (pl.col(belly).mul(2.0)- pl.col(wings[0]) + pl.col(wings[1])).alias(f'mv_neut_{wings[0]}{belly}{wings[1]}_fly'),
        pl.lit([-0.5,1.0,-0.5]).alias('mv_wts')) \
        .with_columns((pl.col(belly).mul(wts[1])- (pl.col(wings[0]) * wts[0] + pl.col(wings[1]) * wts[-1])).alias(
        f'dv01_neut_{wings[0]}{belly}{wings[1]}_fly'), pl.lit(wts).alias('dv01_wts')) \
        .with_columns(
        (pl.col(belly).mul(pca_wts[1])- (pl.col(wings[0]) * pca_wts[0] + pl.col(wings[1]) * pca_wts[-1])).alias(
            f'pca_neut_{wings[0]}{belly}{wings[1]}_fly'),
        pl.lit(pca_wts).alias('pca_wts')) \
        .with_columns(
        (pl.col(belly).mul(pca_wts2[1]) - (pl.col(wings[0]) * pca_wts2[0] + pl.col(wings[1]) * pca_wts2[-1])).alias(
            f'pca_neut2_{wings[0]}{belly}{wings[1]}_fly'),
        pl.lit(pca_wts2).alias('pca_wts'))
    return rates
