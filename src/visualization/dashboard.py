import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict
import logging

class TrustGaugeDashboard:
    """Interactive dashboard for visualizing trust analysis results."""
    
    def __init__(self, config: Dict):
        """
        Initialize the dashboard.
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.theme = config['plotly_theme']
        self.color_palette = config['color_palette']
        self.max_items = config['max_items_per_plot']
    
    def _create_trust_score_plot(self, df):
        """Create trust score distribution plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Product Trust Scores', 'Review Trust Distribution')
        )
        
        # Product trust scores
        product_scores = df.groupBy("product_id", "product_title").agg({
            "product_trust_score": "avg"
        }).toPandas().sort_values("avg(product_trust_score)", ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=product_scores["product_title"].head(self.max_items),
                y=product_scores["avg(product_trust_score)"].head(self.max_items),
                name="Product Trust"
            ),
            row=1, col=1
        )
        
        # Review trust distribution
        review_scores = df.select("review_trust_score").toPandas()
        fig.add_trace(
            go.Histogram(
                x=review_scores["review_trust_score"],
                name="Review Trust"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Trust Score Analysis",
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def _create_sentiment_analysis_plot(self, df):
        """Create sentiment analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentiment vs Rating',
                'Sentiment Distribution',
                'Temporal Sentiment Trend',
                'Aspect Sentiment'
            )
        )
        
        # Sentiment vs Rating scatter
        sentiment_rating = df.select(
            "sentiment_score",
            "star_rating"
        ).toPandas()
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_rating["sentiment_score"],
                y=sentiment_rating["star_rating"],
                mode="markers",
                opacity=0.6,
                name="Reviews"
            ),
            row=1, col=1
        )
        
        # Sentiment distribution
        fig.add_trace(
            go.Histogram(
                x=sentiment_rating["sentiment_score"],
                name="Sentiment"
            ),
            row=1, col=2
        )
        
        # Temporal trend
        temporal = df.groupBy("review_date").agg({
            "sentiment_score": "avg"
        }).toPandas().sort_values("review_date")
        
        fig.add_trace(
            go.Scatter(
                x=temporal["review_date"],
                y=temporal["avg(sentiment_score)"],
                mode="lines",
                name="Trend"
            ),
            row=2, col=1
        )
        
        # Aspect sentiment if available
        if "aspect_sentiment" in df.columns:
            aspects = df.select("aspect_sentiment").toPandas()
            fig.add_trace(
                go.Box(
                    y=aspects["aspect_sentiment"],
                    name="Aspects"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Sentiment Analysis Dashboard",
            template=self.theme
        )
        
        return fig
    
    def _create_fake_detection_plot(self, df):
        """Create visualization for fake review detection."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Fake Review Distribution',
                'Suspicious Pattern Analysis'
            )
        )
        
        # Fake review distribution
        fake_dist = df.groupBy("product_id", "product_title").agg({
            "fake_review_ratio": "avg"
        }).toPandas().sort_values("avg(fake_review_ratio)", ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=fake_dist["product_title"].head(self.max_items),
                y=fake_dist["avg(fake_review_ratio)"].head(self.max_items),
                name="Fake Ratio"
            ),
            row=1, col=1
        )
        
        # Suspicious patterns
        patterns = df.filter(df.is_fake).groupBy(
            "sentiment_extremity"
        ).count().toPandas()
        
        fig.add_trace(
            go.Scatter(
                x=patterns["sentiment_extremity"],
                y=patterns["count"],
                mode="markers",
                name="Patterns"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Fake Review Detection Analysis",
            template=self.theme
        )
        
        return fig
    
    def create_product_comparison(self, df, product_ids):
        """Create comparison visualization for selected products."""
        selected = df.filter(df.product_id.isin(product_ids))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trust Scores',
                'Sentiment Analysis',
                'Review Volume',
                'Verification Ratio'
            )
        )
        
        # Trust score comparison
        trust_comp = selected.groupBy(
            "product_id",
            "product_title"
        ).agg({
            "product_trust_score": "avg",
            "sentiment_score": "avg",
            "review_count": "sum",
            "verified_ratio": "avg"
        }).toPandas()
        
        for metric, row, col in [
            ("avg(product_trust_score)", 1, 1),
            ("avg(sentiment_score)", 1, 2),
            ("sum(review_count)", 2, 1),
            ("avg(verified_ratio)", 2, 2)
        ]:
            fig.add_trace(
                go.Bar(
                    x=trust_comp["product_title"],
                    y=trust_comp[metric],
                    name=metric
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=800,
            title="Product Comparison Dashboard",
            template=self.theme
        )
        
        return fig
    
    def launch(self, df):
        """
        Launch the Gradio interface.
        
        Args:
            df: Spark DataFrame with analysis results
        """
        try:
            def update_visualizations(product_ids):
                trust_fig = self._create_trust_score_plot(df)
                sentiment_fig = self._create_sentiment_analysis_plot(df)
                fake_fig = self._create_fake_detection_plot(df)
                comparison_fig = self.create_product_comparison(
                    df,
                    product_ids.split(",")
                ) if product_ids else None
                
                return [
                    trust_fig,
                    sentiment_fig,
                    fake_fig,
                    comparison_fig
                ]
            
            interface = gr.Interface(
                fn=update_visualizations,
                inputs=[
                    gr.Textbox(
                        label="Enter product IDs (comma-separated)",
                        placeholder="e.g., PROD001,PROD002"
                    )
                ],
                outputs=[
                    gr.Plot(label="Trust Score Analysis"),
                    gr.Plot(label="Sentiment Analysis"),
                    gr.Plot(label="Fake Review Detection"),
                    gr.Plot(label="Product Comparison")
                ],
                title="TrustGauge Dashboard",
                description="Interactive visualization of Amazon review analysis",
                theme="default"
            )
            
            self.logger.info("Launching Gradio interface...")
            interface.launch(
                server_name="0.0.0.0",
                server_port=8000
            )
            
        except Exception as e:
            self.logger.error(f"Error launching dashboard: {str(e)}")
            raise 