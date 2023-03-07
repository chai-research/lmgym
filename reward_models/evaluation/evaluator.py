import pandas as pd


class RewardModelComparitor():
    def __init__(self, base_model, reward_models):
        self.base_model = base_model
        self.reward_models = reward_models

    def predict(self, inputs, best_of):
        responses = self.base_model.predict(inputs, best_of)
        results = {}
        for name, model in self.reward_models.items():
            scored_responses = model.predict(inputs, responses)
            results[name] = scored_responses
        return results

    def pprint(self, results, ranked=True):
        pd.set_option('display.max_colwidth', 100)
        scores = self._get_formatted_results(results)
        df = pd.DataFrame.from_dict(scores, orient='index')
        df = df.sort_values(by=df.columns[0], ascending=False)
        if ranked:
            print(df.rank(method='min'))
        else:
            print(df)
        return df

    def _get_formatted_results(self, results):
        scores = {}
        for model, score_list in results.items():
            for score_dict in score_list:
                text = score_dict['text']
                score = score_dict['score']
                if text not in scores:
                    scores[text] = {model: score}
                else:
                    scores[text][model] = score
        return scores
