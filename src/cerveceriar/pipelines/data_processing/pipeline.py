from kedro.pipeline import Pipeline, node
from cerveceriar.pipelines.data_processing.nodes import load_and_process_data, get_top_reviews, plot_top_reviews
from cerveceriar.nodes.data_processing import process_beer_reviews, analyze_data, plot_kde, calculate_statistics, plot_correlation_matrix, filter_reviews

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_and_process_data,
            inputs="beer_reviews",
            outputs="processed_data",
            name="load_and_process_data_node"
        ),
        node(
            func=get_top_reviews,
            inputs="processed_data",
            outputs="top_reviews",
            name="get_top_reviews_node"
        ),
        node(
            func=plot_top_reviews,
            inputs="top_reviews",
            outputs=None,
            name="plot_top_reviews_node"
        ),
        node(
            func=process_beer_reviews,
            inputs="beer_reviews",
            outputs="processed_beer_reviews",
            name="process_beer_reviews_node"
        ),
        node(
            func=analyze_data,
            inputs="beer_reviews",
            outputs="analysis_output",
            name="analyze_data_node"
        ),
        node(
            func=plot_kde,
            inputs="beer_reviews",
            outputs=None,
            name="plot_kde_node"
        ),
        node(
            func=calculate_statistics,
            inputs="beer_reviews",
            outputs="statistics_output",
            name="calculate_statistics_node"
        ),
        node(
            func=plot_correlation_matrix,
            inputs="beer_reviews",
            outputs=None,
            name="plot_correlation_matrix_node"
        ),
        node(
            func=filter_reviews,
            inputs="beer_reviews",
            outputs="filtered_beer_reviews",
            name="filter_reviews_node"
        )
    ])
