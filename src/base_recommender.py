from abc import ABC, abstractclassmethod
from utils.data_loader import DataLoader
from utils.metric_calculator import MetricCalculator
from utils.models import Dataset, RecommendResult

class BaseRecommender(ABC):
    @abstractclassmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self)  -> None:
        movielens = DataLoader(num_users=1000, num_test_items=5, data_path="../data/ml-10M100K/").load()
        recommend_result = self.recommend(movielens)
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=10
        )
        print(metrics)
