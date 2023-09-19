import pandas as pd
import missingno as msn
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.font_manager import FontProperties
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import construct_features
sys.path.append(os.path.abspath('../visualization'))
import visualize


class FeatureAnalysis:
    def __init__(self):
        # Object Instance
        cf = construct_features.ConstructFeatures()
        self.df_info, self.df_rating = cf.construct_features()
        self.visual = visualize.Visualize()

    @staticmethod
    def debug_text(title, task):
        print('\n')
        print("=" * 150)
        print('◘ ', title)

        try:
            print(task)

        except Exception as exc:
            print("! ", exc)

        finally:
            print("=" * 150)

    @staticmethod
    def display_dataframe(name, df, contents):
        df_table = df.head(contents)
        print('\n')
        print("=" * 150)
        print("◘ ", name, " Dataframe:")
        print(df_table.to_string())
        print("=" * 150)

    def feature_analysis(self):
        # Inspect distribution for Ratings
        bins = 10
        d1 = self.df_rating['rating']
        d2 = self.df_rating['book_price']
        title_d1 = "Histogram: Book Ratings"
        title_d2 = "Histogram: Book Prices"
        self.visual.plot_multi_histogram(d1, d2, bins, title_d1, title_d2)

        # Density Inspection
        df = self.df_rating['rating']
        text = "Kde Plot: Book Ratings"
        x_label = "Ratings"
        y_label = "Frequency"
        self.visual.plot_kde(df, text, x_label, y_label)    # !!!

        df = self.df_rating['book_price']
        text = "Kde Plot: Book Prices"
        x_label = "Prices"
        y_label = "Frequency"
        self.visual.plot_kde(df, text, x_label, y_label)    # !!!

        # Pearson Correlation for Numerical features
        df_heatmap = self.df_rating[['rating', 'book_price']].copy()
        self.visual.plot_correlation(df_heatmap)

        # Acquire top 10 Book Genres
        text = "Pie Distribution: Book Genres"
        genre = self.df_info['categories'].value_counts().sort_values(ascending=False)
        genre = genre.head(10)
        self.debug_text(text, genre)

        # Inspect distribution for Genres
        bbox_to_anchor = (1, 1.2)
        labels = genre.keys().map(str)
        explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.visual.plot_pie(genre, explode, labels, text, bbox_to_anchor)

        kind = 'barh'
        text = "Bar Chart: Book Genres"
        x_label = "Frequency"
        y_label = "Genres"
        self.visual.plot_dataframe(genre, text, kind, x_label, y_label)

        # Identify how much Book Ratings affect its Prices
        ratings_price = self.df_rating[['rating', 'book_price']]
        ratings_price = ratings_price.sort_values(by=['book_price', 'rating'], ascending=False)
        self.display_dataframe("Ratings vs Price", ratings_price, 20)

        x = self.df_rating['rating']
        y = self.df_rating['book_price']

        text = "Scatter Plot: Ratings and Price"
        x_label = 'Ratings'
        y_label = 'Book Price'
        self.visual.plot_scatter(x, y, text, x_label, y_label)

        # Books most purchased by users
        most_purchases = self.df_rating.groupby('book_title')['user_id'].count().sort_values()
        df_temp_rating = most_purchases.to_frame()
        df_temp_rating['most_purchases'] = most_purchases

        df_arg = df_temp_rating['most_purchases'].sort_values(ascending=False)
        self.display_dataframe("Books most purchased", df_arg, 15)

        # Inspect query from the visualization chart
        x = most_purchases.values[-15:]
        y = most_purchases.index[-15:]
        text = "Bar Plot: Books most bought purchased"
        x_label = "Purchases"
        y_label = "Books"
        self.visual.plot_plotly_bar(x, y, text, x_label, y_label)

        # Highest Rated Books (Mean)
        highest_rated = self.df_rating.groupby('book_title')['rating'].mean()
        df_temp_rating = highest_rated.to_frame()
        df_temp_rating['mean_ratings'] = highest_rated

        df_arg = df_temp_rating['mean_ratings'].sort_values(ascending=False)
        self.display_dataframe("Highest Rated Books", df_arg, 15)

        # Inspect query from the visualization chart
        x = highest_rated.values[-15:]
        y = highest_rated.index[-15:]
        text = "Bar Plot: Highest Rated Books"
        x_label = "Ratings"
        y_label = "Books"
        self.visual.plot_plotly_bar(x, y, text, x_label, y_label)

        # Expensive Books in store (highest mean Price)
        expensive_books = self.df_rating.groupby('book_title')['book_price'].mean()
        df_temp_rating = expensive_books.to_frame()
        df_temp_rating['mean_price'] = expensive_books

        df_arg = df_temp_rating['mean_price'].sort_values(ascending=False)
        self.display_dataframe('Top Expensive Books', df_arg, 15)

        # Distribution of Mean Book Prices
        text = "Histogram: Book Price distribution"
        x_label = "Price Range ($)"
        y_label = "Frequency"
        kind = 'hist'
        self.visual.plot_dataframe(df_arg, text, kind, x_label, y_label)

        # Top-rated Books accumulating over 3500 Ratings in total (per book)
        accumulated_ratings = self.df_info[self.df_info['ratings_count'] > 3500][['book_title', 'ratings_count']]\
            .drop_duplicates()
        df_arg = accumulated_ratings.sort_values(by=['ratings_count'], ascending=False)
        self.display_dataframe("Books over 3500 Ratings", df_arg, 15)

        # Generate a Bar Plot for visual evidence
        text = "Bar Plot: Books over 3500 Ratings"
        x = accumulated_ratings['ratings_count']
        y = accumulated_ratings['book_title']
        x_label = "Ratings"
        y_label = "Books"
        self.visual.plot_plotly_bar(x, y, text, x_label, y_label)

        # Aggregate books for a particular category
        category_books = self.df_info.groupby('categories')['book_title'].count().sort_values()
        df_temp_info = category_books.to_frame()
        df_temp_info['category_books'] = category_books

        df_arg = category_books.sort_values(ascending=False)
        self.display_dataframe("15 Top Books in a Category", df_arg, 15)

        # Inspect query from the visualization chart
        text = "Bar Plot: Top 15 Categorical Books"
        x_label = "Books"
        y_label = "Categories"
        self.visual.plot_bar(df_arg, text, x_label, y_label, 'h')

        # Authors with the most published books
        author_publish = self.df_info.groupby('book_author')['book_title'].count().sort_values().sort_values()
        df_temp_info = author_publish.to_frame()
        df_temp_info['author_publish'] = author_publish

        df_arg = author_publish.sort_values(ascending=False)
        self.display_dataframe("Most Books by Author", df_arg, 15)

        # Represent the query via a Bar chart
        text = "Bar Plot: Most Published Books by Authors"
        x_label = "Authors"
        y_label = "Publishes"
        self.visual.plot_bar(df_arg, text, x_label, y_label, 'h')

        # Author's active years
        author_years = self.df_info.groupby('book_author')['published_year'].nunique()
        df_temp_info = author_years.to_frame()
        df_temp_info['author_years'] = author_years

        df_arg = author_years.sort_values(ascending=False)
        self.display_dataframe("Years most active by Authors", df_arg, 15)

        # Inspect the distribution for the most active years
        subtext = "Top 15 Active Authors"
        df_arg = df_arg.head(15)
        bbox_to_anchor = (1, 1.2)
        labels = df_arg.keys().map(str)
        explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.visual.plot_pie(df_arg, subtext, labels, explode, bbox_to_anchor)

        # CONTINUE  2.12 Notebook
        # Authors working with multiple genres
        author_categories = self.df_info.groupby('book_author')['categories'].nunique()
        df_temp_info = author_categories.to_frame()
        df_temp_info['author_categories'] = author_categories

        df_arg = author_categories.sort_values(ascending=False)
        self.display_dataframe("Authors with diverse Categories", df_arg, 15)

        # Display all analyzed Dataframes
        self.display_dataframe("Info data", df_temp_info, 25)
        self.display_dataframe("Rating data", df_temp_rating, 25)


if __name__ == "__main__":
    main = FeatureAnalysis()
    main.feature_analysis()
