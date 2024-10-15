from pathlib import Path
import pandas as pd
import sys
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PARENT_DIR))

import os


class Overview:

    def __init__(self):
        self.data_path = os.path.join(ROOT_DIR, "data")

    def read_data(self):
        """Reads the three csvs that have users table, emotional_data table and loan_table
        and returns Dataframe with the data"""
        df_users = pd.read_csv(os.path.join(self.data_path, "user_cleaned_data.csv"))
        df_loans = pd.read_csv(os.path.join(self.data_path, "loans.csv"))
        df_emotional_data = pd.read_csv(os.path.join(self.data_path, "final_emotional_data.csv"))
        return df_users, df_loans, df_emotional_data

    #

    def load_page(self):
        """Loads all the charts used in visualization"""
        def draw_emotion_bar_chart(data, x_column, y_column, color_column, x_label, y_label):
            """
            Draw a bar chart using Plotly and display it in a Streamlit container.

            Args:
            - data (DataFrame): The data to plot.
            - x_column (str): The column to be used on the x-axis.
            - y_column (str): The column to be used on the y-axis.
            - color_column (str): The column used to define the color grouping.
            - x_label (str): Label for the x-axis.
            - y_label (str): Label for the y-axis.
            """
            # Plotting with Plotly Express
            fig = px.bar(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                labels={x_column: x_label, y_column: y_label},
                barmode='group',  # To have clustered bars
                height=600,
                width=900,
            )

            # Update layout for better readability
            fig.update_layout(
                xaxis_tickangle=90,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title='Emotions',
                xaxis={"categoryorder": "total descending"}
            )

            # Display the chart inside a container
            with st.container():
                st.plotly_chart(fig)

        def plot_scatter(df, x_column, y_column, title, x_label, y_label, cmap='coolwarm', alpha=0.6, figsize=(12, 6)):
            """
            Creates a scatter plot and displays it using Streamlit.

            Args:
            -df: DataFrame
            - x_column: Column name for x-axis data.
            - y_column: Column name for y-axis data.
            - title: Title of the plot.
            - x_label: Label for the x-axis.
            - y_label: Label for the y-axis.
            - cmap: Colormap for the scatter plot (default is 'coolwarm').
            - alpha: Transparency level for the scatter points (default is 0.6).
            - figsize: Figure size for the plot (default is (12, 6)).

            Returns:
            - Scatter plot displayed in Streamlit.
            """
            plt.figure(figsize=figsize)

            # Create scatter plot
            scatter = plt.scatter(df[x_column], df[y_column], cmap=cmap, alpha=alpha)

            # Add title and labels
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            # Display plot
            st.pyplot(plt)

        #data preprocessing #convert all date columns to timestamps
        def convert_to_timestamp(df, *columns):

            for column in columns:
                if column in df.columns:
                    df[column] = pd.to_datetime(df[column])
            return df

        convert_to_timestamp(df_loans, "issue_date", "due_date", "paid_date")

        #Building the Dashboard
        st.header("Emphathic Credit System Dashboard", divider="rainbow")
        st.subheader("KPI Metrics")

        #Read the data
        dataloader = Overview()

        df_users, df_loans, df_emotional_data = dataloader.read_data()
        #Get total profit
        df_loans["profit"] = df_loans["loan_amount_paid"] - df_loans["loan_amount"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            #get the unique users from the users_table
            st.write("Unique Users")
            st.subheader(f"{df_users.shape[0]}")
        with col2:
            #get the total loans given out over the years
            st.write("Total Loans Disbursed")
            st.subheader(f"{df_loans['loan_amount'].sum():,}")

        with col3:
            #get the total amount of profit over the years
            st.write("Total Profit")
            total_profit = df_loans['profit'].sum()
            st.subheader(f"{total_profit:,.0f}")

        with col4:
            #get the average percentage of profit across the years
            st.write("Average % Profit")
            df_loans["percentage_profit"] = df_loans["profit"] / df_loans[
                "loan_amount"] * 100
            average_profit = df_loans["percentage_profit"].mean()
            st.subheader(f"{average_profit:,.0f}")

        st.divider()

        ##Load Emotional Data and Analyze
        #convert timestamp column to timestamp
        df_emotional_data = convert_to_timestamp(df_emotional_data, "timestamp")

        # fill all the nulls with "Unknown"
        df_emotional_data = df_emotional_data.fillna("Unknown")

        #Extract the year
        df_emotional_data["Year"] = df_emotional_data["timestamp"].dt.year

        # get the number of primary_emotion per year
        #preprocess the primary_emotion into negative and positive emotions

        # Categorize emotions into positive and negative
        positive_emotions = [
            'joy', 'love', 'pride', 'anticipation', 'amusement',
            'trust', 'relief', 'surprise'
        ]
        negative_emotions = [
            'sadness', 'boredom', 'shame', 'contempt', 'guilt',
            'confusion', 'frustration', 'disgust', 'jealousy', 'anxiety', 'fear'
        ]

        df_emotional_data['emotion_category'] = df_emotional_data['primary_emotion'].apply(
            lambda x: "Positive" if x in positive_emotions else "Negative"
        )

        # Categorize intensity as high or low
        df_emotional_data["intensity_category"] = df_emotional_data["intensity"].apply(
            lambda x: "Low" if 0.0 <= x <= 5.0 else "High")

        #Combine intensity_category and emotion_category
        df_emotional_data["emotional_intensity"] = df_emotional_data['intensity_category'] + '-' + df_emotional_data[
            'emotion_category']

        #Plot the charts to show emotional trends over the years
        st.subheader("Emotional Trends Over the years")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Analyzing Primary Emotions over the Years")
            # Draw Emotional_Category
            dominant_emotion = df_emotional_data.groupby(["Year", "emotion_category"]).size().reset_index(
                name="Count").sort_values(by=["Year", "Count"], ascending=False)

            draw_emotion_bar_chart(dominant_emotion, 'Year', 'Count', 'emotion_category',
                                   'Year', 'Count')

        with col3:
            #Analyze the intensity distribution
            # Draw Distribution on Intensity
            st.write("Distribution of Intensity")

            # histogram of intensity
            fig = px.histogram(
                df_emotional_data,
                x="intensity",
                nbins=19,  # Number of bins in the histogram
            )

            # Display the histogram in Streamlit
            st.plotly_chart(fig)

        with col2:

            ## Group by Year and emotional_intensity and count occurrences
            st.write("Analyzing Emotions and Intensity Combination")
            dominant_emotion_intensity = df_emotional_data.groupby(["Year", "emotional_intensity"]).size().reset_index(
                name="Count").sort_values(by=["Year", "Count"], ascending=False)

            # Draw Emotional_Intensity Chart
            draw_emotion_bar_chart(dominant_emotion_intensity, 'Year', 'Count', 'emotional_intensity',
                                   'Year', 'Count')

        # List of categorical columns to visualize
        st.subheader("Analyzing how different emotions are affected by context.")
        categorical_columns = ['relationship', 'situation', 'time_of_day']

        # Create three columns for layout
        col1, col2, col3 = st.columns(3)

        # Step 1 & 2: Loop through each categorical column and group them to the emotion category
        for i, column in enumerate(categorical_columns):
            # Determine which column to use based on the index
            if i % 3 == 0:
                current_col = col1
            elif i % 3 == 1:
                current_col = col2
            else:
                current_col = col3

            # Group emotion_category according to the categorical variables
            top_emotions = df_emotional_data.groupby(column)['emotion_category'].value_counts().reset_index(
                name='count')

            with current_col:
                draw_emotion_bar_chart(top_emotions, x_column=column, y_column='count', color_column='emotion_category',
                                       x_label=column, y_label='count')

        ##ANALYZE EMOTIONAL PATTERNS AND LOAN TERMS
        st.subheader("Analyzing emotional patterns and loan terms")
        col1, col2 = st.columns(2)
        #Get the average Emotional Grade Score
        yearly_average_grade = df_emotional_data.groupby(["Year", "user_id", ]).agg(
            grade=("grade", "mean"),
            intensity=("intensity", "mean")
        ).reset_index()

        # create year from the date loans were issued
        df_loans["Year"] = df_loans["issue_date"].dt.year

        # aggregate total amount of loans per year
        yearly_loan_amount = df_loans.groupby(["Year", "user_id"])["loan_amount"].sum().reset_index()

        # merge the two tables together on user_id and Year
        df_loans_emotions_merged = pd.merge(yearly_average_grade, yearly_loan_amount, on=["user_id", "Year"],
                                            how="left")
        # drop all nulls in loan amount
        df_loans_emotions_merged.dropna(inplace=True)

        # merge the two tables together
        df_loans_emotions_merged = pd.merge(yearly_average_grade, yearly_loan_amount, on=["user_id", "Year"],
                                            how="left")

        # remove all the nulls in the table
        df_loans_emotions_merged.dropna(inplace=True)

        # create a scatter plot to look at relationship between average grade score and loan amount
        with col1:
            plot_scatter(df_loans_emotions_merged, 'grade', 'loan_amount',
                         cmap="coolwarm", alpha=0.6, title="Scatter plot of Emotional Grade vs Loan Amount",
                         x_label="Emotion_Grade",
                         y_label="Loan Amount"
                         )

        # Analyze Grade vs Interest rate
        with col2:
            # add interest rate and credit_limit from users table
            df_loans_emotions_merged = pd.merge(df_loans_emotions_merged,
                                                df_users[["user_id", "interest_rate", "loan_term", "credit_limit"]],
                                                on="user_id",
                                                how="left")

            # create a scatter plot to look at relationship between grade score and Interest Rate
            plot_scatter(df_loans_emotions_merged, 'grade', 'interest_rate',
                         cmap="coolwarm", alpha=0.6, title="Scatter plot of Emotional Grade vs Interest Rate",
                         x_label="Emotion_Grade",
                         y_label="Interest Rate"
                         )

        col1, col2 = st.columns(2)
        with col1:
            # create a scatter plot to look at relationship between intensity and loan amount
            plot_scatter(df_loans_emotions_merged, 'intensity', 'loan_amount',
                         cmap="coolwarm", alpha=0.6, title="Scatter plot of Emotional Intensity vs Loan Amount",
                         x_label="Loan Amount",
                         y_label="Interest Rate"
                         )
        with col2:
            plot_scatter(df_loans_emotions_merged, 'intensity', 'interest_rate',
                         cmap="coolwarm", alpha=0.6, title="Scatter plot of Emotional Intensity vs Interest Rate",
                         x_label="Intensity",
                         y_label="Interest Rate"
                         )

        col1, col2 = st.columns(2)

        with col1:
            # create a scatter plot to look at relationship between intensity and credit limit
            plot_scatter(df_loans_emotions_merged, 'intensity', 'credit_limit',
                         cmap="coolwarm", alpha=0.6, title="Scatter plot of Emotional Intensity vs Credit Limit",
                         x_label="Intensity",
                         y_label="Credit Limit"
                         )

        with col2:
            # create a scatter plot to look at relationship between grade and credit limit
            plot_scatter(df_loans_emotions_merged, 'grade', 'credit_limit',
                         cmap="coolwarm", alpha=0.6, title="Scatter plot of Emotional Grade vs Credit Limit",
                         x_label="Intensity",
                         y_label="Credit Limit"

                         )
        # Calculate correlation matrix for all columns in merged dataframe that has loan information and emotional
        # intensity and emotional grade
        corr_matrix = df_loans_emotions_merged.corr()
        # Create a matplotlib figure for the heatmap
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)

        # Add title to the heatmap
        ax.set_title('Correlation of emotional factors and loan terms')

        with st.container():
            # Display the heatmap in the Streamlit app
            st.pyplot(fig)

        ###ANALYZE LOAN PERFORMANCE
        # merge with users data to analyze Loan Performance
        st.subheader("Loan Performance")
        df_loans_merged = pd.merge(df_loans, df_users[["interest_rate", "loan_term", "user_id"]], on="user_id",
                                   how="left")
        # group per year
        loan_data_grouped = df_loans_merged.groupby("Year")[["profit", "loan_amount"]].sum().reset_index()

        loan_data_grouped.head()
        #Categorize year by the year that the loan was given and categorize year by the paid date
        df_loans_merged["Year_Loan_Given"] = df_loans["issue_date"].dt.year
        df_loans_merged["Year_Loan_Paid_Back"] = df_loans["paid_date"].dt.year

        #Total Amount of Money loaned out per year
        yearly_loan_payouts = df_loans_merged.groupby("Year")["loan_amount"].sum().reset_index()
        yearly_loans_paid_back = df_loans_merged.groupby("Year")["loan_amount_paid"].sum().reset_index()
        #Merge both to get the profit per year
        df_merged = pd.merge(yearly_loan_payouts, yearly_loans_paid_back, on="Year")
        df_merged["profit"] = df_merged["loan_amount_paid"] - df_merged["loan_amount"]

        #Analyzing profits and default rates

        ##Define bins of loan categories
        # Define the bins (loan ranges) and their corresponding labels
        bins = [0, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000]
        labels = ['0-2k', '2k-4k', '4k-6k', '6k-8k', '8k-10k', "10k-20k", "20k-30k", "30k-40k", "40k-50k"]

        # Create a new column 'loan_amount_category' based on these bins
        df_loans_merged['loan_amount_category'] = pd.cut(df_loans_merged['loan_amount'], bins=bins, labels=labels,
                                                         right=False)

        # percentage of defaulters
        defaulters = df_loans_merged[df_loans_merged["status"] == "late"]
        defaulters_per_loan_category = defaulters.groupby("loan_amount_category").size().reset_index(name="count")

        col1, col2 = st.columns(2)
        with col1:
            # Create a line chart using Plotly Graph Objects
            fig = go.Figure()
            # Add profit line
            fig.add_trace(go.Scatter(
                x=df_merged['Year'],
                y=df_merged['profit'],
                mode='lines+markers+text',
                name='Profit',
                text=loan_data_grouped['profit'].round(0).astype(str),
                textposition='top center',  # Position text on the line
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))

            # Add loan amount line
            fig.add_trace(go.Scatter(
                x=df_merged['Year'],
                y=df_merged['loan_amount'],
                mode='lines+markers+text',
                name='Loan Amount',
                text=loan_data_grouped['loan_amount'].astype(str),
                textposition='top center',  # Position text on the line
                line=dict(color='orange', width=2),
                marker=dict(size=8)
            ))

            # Update layout with titles
            fig.update_layout(
                title='Total Profit and Loan Amount Disbursed per Year',
                xaxis_title='Year',
                yaxis_title='Amount',
                legend_title='Metric'
            )

            # Show the plot
            st.plotly_chart(fig)
            #

        with col2:
            # Aggregate profit by year and loan_amount category
            df_loans_merged.groupby(["loan_amount_category", "Year_Loan_Paid_Back"])[
                "profit"].sum().reset_index().sort_values(
                by="profit", ascending=False)

            # plot bar chart
            fig = px.bar(
                df_loans_merged.groupby(["loan_amount_category", "Year_Loan_Paid_Back"])["profit"].sum().reset_index(),
                x="Year_Loan_Paid_Back",
                y="profit",
                color="loan_amount_category",
                barmode="group",
                title="Total Profit by Loan Amount Category and Year",
                height=600,
                width=1000
            )

            # Update layout for better readability
            fig.update_layout(
                xaxis_title="Year",
                legend_title='Category of Loans Given',
                xaxis={"categoryorder": "total descending"}
            )

            st.plotly_chart(fig)

        ##Define bins of loan categories
        # Define the bins (loan ranges) and their corresponding labels
        bins = [0, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000]
        labels = ['0-2k', '2k-4k', '4k-6k', '6k-8k', '8k-10k', "10k-20k", "20k-30k", "30k-40k", "40k-50k"]

        # Create a new column 'loan_amount_category' based on these bins
        df_loans_merged['loan_amount_category'] = pd.cut(df_loans_merged['loan_amount'], bins=bins, labels=labels,
                                                         right=False)

        #percentage of defaulters in each category
        defaulters = df_loans_merged[df_loans_merged["status"] == "late"]
        defaulters_per_loan_category = defaulters.groupby("loan_amount_category").size().reset_index(name="count")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(defaulters_per_loan_category,
                         x='loan_amount_category',
                         y='count',
                         title='Defaulters per Loan Amount Category',
                         labels={'loan_amount_category': 'Loan Amount Category'},
                         )

            st.plotly_chart(fig)

        with col2:
            ##By interest rate
            defaulters_per_interest_rate = defaulters.groupby("interest_rate").size().reset_index(name="count")
            fig = px.pie(defaulters_per_interest_rate,
                         names='interest_rate',
                         values='count',
                         title='Defaulters per Interest Rate Group',
                         labels={'interest_rate_group': 'Interest Rate Group'},
                         hole=0.4)

            # Add text labels to show both label and percentage inside the pie chart
            fig.update_traces(textinfo='label+percent',
                              textfont_size=12,  # Adjust font size
                              textposition='outside')  # Ensure text is inside the slices

            st.plotly_chart(fig)
