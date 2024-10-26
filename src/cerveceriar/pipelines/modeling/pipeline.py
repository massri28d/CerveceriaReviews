from kedro.pipeline import Pipeline, node
from .nodes import categorize_review, split_features_target, split_data, train_decision_tree, evaluate_model, visualize_tree

def create_pipeline(**kwargs):
    print("Creando pipeline...")
    pipeline = Pipeline(
        [
            node(
                func=categorize_review,
                inputs="raw_data",
                outputs="data_categorized",
                name="categorize_review_node",
            ),
            node(
                func=split_features_target,
                inputs="data_categorized",
                outputs=["X", "y"],
                name="split_features_target_node",
            ),
            node(
                func=split_data,
                inputs=["X", "y"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_decision_tree,
                inputs=["X_train", "y_train"],
                outputs="clf",
                name="train_decision_tree_node",
            ),
            node(
                func=evaluate_model,
                inputs=["clf", "X_test", "y_test"],
                outputs=["accuracy", "classification_report"],
                name="evaluate_model_node",
            ),
            node(
                func=visualize_tree,
                inputs=["clf", "params:feature_names"],
                outputs=None,
                name="visualize_tree_node",
            ),
        ]
    )
    print("Pipeline creada.")
    return pipeline

