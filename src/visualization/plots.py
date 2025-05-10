import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class TrustGaugeVisualizer:
    def __init__(self):
        self.color_scale = 'RdYlGn'
        self.template = 'plotly_white'

    def plot_top_categories(self, df, top_n=10):
        """Plot top product categories by review count"""
        top_categories = df.groupBy("product_category").count() \
            .orderBy("count", ascending=False) \
            .limit(top_n) \
            .toPandas()

        fig = px.bar(
            top_categories,
            x='product_category',
            y='count',
            title=f'Top {top_n} Product Categories by Review Count',
            labels={'product_category': 'Category', 'count': 'Number of Reviews'},
            color='count',
            color_continuous_scale=self.color_scale
        )
        fig.update_layout(xaxis_tickangle=-45, template=self.template)
        return fig

    def plot_rating_distribution(self, df):
        """Plot distribution of star ratings"""
        rating_dist = df.groupBy("star_rating").count() \
            .orderBy("star_rating") \
            .toPandas()

        total = rating_dist['count'].sum()
        rating_dist['percentage'] = (rating_dist['count'] / total * 100).round(1)

        fig = px.bar(
            rating_dist,
            x='star_rating',
            y='count',
            title=f'Distribution of Star Ratings (Total: {total:,} Reviews)',
            labels={'star_rating': 'Star Rating', 'count': 'Number of Reviews'},
            color='star_rating',
            color_continuous_scale=self.color_scale,
            text=rating_dist['percentage'].apply(lambda x: f'{x}%')
        )
        fig.update_layout(template=self.template)
        return fig

    def plot_trust_score_distribution(self, df):
        """Plot distribution of trust scores"""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Product Trust Scores", "Company Trust Scores"))

        # Product trust scores
        fig.add_trace(
            go.Histogram(x=df.select("final_product_score").toPandas()["final_product_score"],
                        name="Product Scores"),
            row=1, col=1
        )

        # Company trust scores
        fig.add_trace(
            go.Histogram(x=df.select("final_company_score").toPandas()["final_company_score"],
                        name="Company Scores"),
            row=1, col=2
        )

        fig.update_layout(
            title_text="Trust Score Distribution",
            template=self.template,
            showlegend=True
        )
        return fig

    def plot_category_trust_scores(self, df, top_n=15):
        """Plot average trust scores by category"""
        category_scores = df.groupBy("product_category") \
            .agg(
                {"final_product_score": "mean", "final_company_score": "mean"}
            ) \
            .orderBy("avg(final_product_score)", ascending=False) \
            .limit(top_n) \
            .toPandas()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=category_scores['product_category'],
            y=category_scores['avg(final_product_score)'],
            name='Product Trust Score'
        ))
        fig.add_trace(go.Bar(
            x=category_scores['product_category'],
            y=category_scores['avg(final_company_score)'],
            name='Company Trust Score'
        ))

        fig.update_layout(
            title=f'Average Trust Scores by Category (Top {top_n})',
            xaxis_title='Category',
            yaxis_title='Trust Score',
            template=self.template,
            barmode='group'
        )
        return fig

    def plot_pipeline_metrics(self, initial_count, standardized_count, final_count):
        """Plot pipeline metrics showing data reduction at each stage"""
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number",
            value=initial_count,
            number={'valueformat': ',d'},
            title={"text": "🟢  Original Total Rows"},
            domain={'row': 0, 'column': 0}
        ))
        fig.add_trace(go.Indicator(
            mode="number",
            value=standardized_count,
            number={'valueformat': ',d'},
            title={"text": "🛠  After Data Standardization"},
            domain={'row': 0, 'column': 1}
        ))
        fig.add_trace(go.Indicator(
            mode="number",
            value=final_count,
            number={'valueformat': ',d'},
            title={"text": "✅  After Data Quality Check"},
            domain={'row': 0, 'column': 2}
        ))
        fig.update_layout(
            grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
            template=self.template,
            title="📊  Row Counts at Each Pipeline Stage",
            margin=dict(t=50, l=20, r=20, b=20)
        )
        return fig 