import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Student Performance Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @keyframes glow {
        0% { text-shadow: 0 0 10px rgba(255,0,0,0.5); }
        50% { text-shadow: 0 0 20px rgba(255,0,0,0.8); }
        100% { text-shadow: 0 0 10px rgba(255,0,0,0.5); }
    }
    
    .stApp {
        background: linear-gradient(to right, #000000, #1a1a1a, #000000);
        color: #ffffff;
    }
    
    .stSidebar {
        background: rgba(0, 0, 0, 0.9);
        border-right: 2px solid #ff0000;
    }
    
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid #ff0000;
    }
    
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #ff0000 0%, #cc0000 100%);
    }
    
    .glow-text {
        color: #ff0000;
        animation: glow 2s infinite;
    }

    .stMultiSelect [data-baseweb="select"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid #ff0000 !important;
        color: white !important;
    }

    .stMultiSelect [data-baseweb="select"]:hover {
        border-color: #cc0000 !important;
        box-shadow: 0 0 10px rgba(255,0,0,0.3) !important;
    }

    .stMultiSelect [data-baseweb="popover"] {
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid #ff0000 !important;
    }

    .stMultiSelect [data-baseweb="option"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: white !important;
    }

    .stMultiSelect [data-baseweb="option"]:hover {
        background-color: rgba(255,0,0,0.3) !important;
    }

    .stCheckbox [data-baseweb="checkbox"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid #ff0000 !important;
    }

    .stCheckbox [data-baseweb="checkbox"]:hover {
        border-color: #cc0000 !important;
        box-shadow: 0 0 10px rgba(255,0,0,0.3) !important;
    }

    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] {
        background-color: #ff0000 !important;
        border-color: #ff0000 !important;
    }

    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid #ff0000 !important;
        color: white !important;
    }

    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #cc0000 !important;
        box-shadow: 0 0 10px rgba(255,0,0,0.3) !important;
    }

    .stSelectbox [data-baseweb="popover"] {
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid #ff0000 !important;
    }

    .stSelectbox [data-baseweb="option"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: white !important;
    }

    .stSelectbox [data-baseweb="option"]:hover {
        background-color: rgba(255,0,0,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Student_performance_data .csv")
    return df

df = load_data()

st.markdown("<h1 class='glow-text' style='text-align: center;'>Student Performance Analytics Dashboard</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='color: #ff0000;'>Filters</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #ff0000; font-size: 1.1em;'>Academic Performance</h3>", unsafe_allow_html=True)
    min_gpa = float(df['GPA'].min())
    max_gpa = float(df['GPA'].max())
    gpa_range = st.slider("GPA Range", min_gpa, max_gpa, (min_gpa, max_gpa))
    
    st.markdown("<h3 style='color: #ff0000; font-size: 1.1em;'>Study Habits</h3>", unsafe_allow_html=True)
    min_study = float(df['StudyTimeWeekly'].min())
    max_study = float(df['StudyTimeWeekly'].max())
    study_range = st.slider("Weekly Study Time (hours)", min_study, max_study, (min_study, max_study))
    
    st.markdown("<h3 style='color: #ff0000; font-size: 1.1em;'>Support Services</h3>", unsafe_allow_html=True)
    tutoring_options = st.multiselect(
        "Tutoring Status",
        options=["With Tutoring", "Without Tutoring"],
        default=["With Tutoring", "Without Tutoring"]
    )
    
    st.markdown("<h3 style='color: #ff0000; font-size: 1.1em;'>Parental Involvement</h3>", unsafe_allow_html=True)
    parental_levels = st.multiselect(
        "Parental Support Level",
        options=[f"Level {i}" for i in sorted(df['ParentalSupport'].unique())],
        default=[f"Level {i}" for i in sorted(df['ParentalSupport'].unique())]
    )
    
    st.markdown("<h3 style='color: #ff0000; font-size: 1.1em;'>Extracurricular Activities</h3>", unsafe_allow_html=True)
    activities = ['Extracurricular', 'Sports', 'Music', 'Volunteering']
    selected_activities = {}
    for activity in activities:
        selected_activities[activity] = st.checkbox(activity, value=True)
    
    st.markdown("<h3 style='color: #ff0000; font-size: 1.1em;'>Attendance</h3>", unsafe_allow_html=True)
    min_absences = float(df['Absences'].min())
    max_absences = float(df['Absences'].max())
    absences_range = st.slider("Absences Range", min_absences, max_absences, (min_absences, max_absences))

filtered_df = df[
    (df['GPA'].between(gpa_range[0], gpa_range[1])) &
    (df['StudyTimeWeekly'].between(study_range[0], study_range[1])) &
    (df['Absences'].between(absences_range[0], absences_range[1])) &
    (df['ParentalSupport'].isin([int(level.split()[-1]) for level in parental_levels])) &
    (
        ((df['Tutoring'] == 1) & ("With Tutoring" in tutoring_options)) |
        ((df['Tutoring'] == 0) & ("Without Tutoring" in tutoring_options))
    )
]

for activity, is_selected in selected_activities.items():
    if not is_selected:
        filtered_df = filtered_df[filtered_df[activity] == 0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Average GPA",
        value=f"{filtered_df['GPA'].mean():.2f}"
    )

with col2:
    st.metric(
        label="Average Study Time",
        value=f"{filtered_df['StudyTimeWeekly'].mean():.1f} hrs"
    )

with col3:
    st.metric(
        label="Total Students",
        value=len(filtered_df)
    )

col1, col2 = st.columns(2)

with col1:
    fig_gpa = px.histogram(
        filtered_df, x='GPA',
        title='GPA Distribution',
        nbins=20,
        color_discrete_sequence=['#ff0000']
    )
    fig_gpa.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#ff0000',
        xaxis_title="GPA",
        yaxis_title="Count",
        xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
        yaxis=dict(gridcolor='rgba(255,0,0,0.2)')
    )
    st.plotly_chart(fig_gpa, use_container_width=True)

with col2:
    fig_scatter = px.scatter(
        filtered_df, x='StudyTimeWeekly', y='GPA',
        title='Study Time vs GPA',
        color='Tutoring',
        color_continuous_scale='Reds'
    )
    fig_scatter.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#ff0000',
        xaxis_title="Weekly Study Time (hours)",
        yaxis_title="GPA",
        xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
        yaxis=dict(gridcolor='rgba(255,0,0,0.2)')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

col_circle1, col_circle2 = st.columns(2)

with col_circle1:
    parental_counts = filtered_df['ParentalSupport'].value_counts()
    fig_parental_donut = go.Figure(data=[go.Pie(
        labels=[f'Level {i}' for i in parental_counts.index],
        values=parental_counts.values,
        hole=.4,
        marker_colors=['#ff0000', '#cc0000', '#990000', '#660000']
    )])
    fig_parental_donut.update_layout(
        template='plotly_dark',
        title=dict(
            text='Parental Support Distribution',
            font=dict(color='#ff0000', size=20),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(color='#ffffff'),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig_parental_donut, use_container_width=True)

with col_circle2:
    activity_counts = [filtered_df[activity].sum() for activity in activities]
    fig_activities_donut = go.Figure(data=[go.Pie(
        labels=activities,
        values=activity_counts,
        hole=.4,
        marker_colors=['#ff0000', '#cc0000', '#990000', '#660000']
    )])
    fig_activities_donut.update_layout(
        template='plotly_dark',
        title=dict(
            text='Activity Participation',
            font=dict(color='#ff0000', size=20),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(color='#ffffff'),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig_activities_donut, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig_parental = px.box(
        filtered_df, x='ParentalSupport', y='GPA',
        title='Parental Support Impact on GPA'
    )
    fig_parental.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#ff0000',
        xaxis_title="Parental Support Level",
        yaxis_title="GPA",
        xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
        yaxis=dict(gridcolor='rgba(255,0,0,0.2)')
    )
    st.plotly_chart(fig_parental, use_container_width=True)

with col4:
    fig_absences = px.scatter(
        filtered_df, x='Absences', y='GPA',
        title='Absences Impact on GPA',
        trendline="ols"
    )
    fig_absences.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#ff0000',
        xaxis_title="Number of Absences",
        yaxis_title="GPA",
        xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
        yaxis=dict(gridcolor='rgba(255,0,0,0.2)')
    )
    st.plotly_chart(fig_absences, use_container_width=True)

st.markdown("<h3 style='color: #ff0000;'>Summary Statistics</h3>", unsafe_allow_html=True)
summary_stats = filtered_df.describe()
st.dataframe(
    summary_stats,
    use_container_width=True,
    hide_index=True
)

st.markdown("<h3 style='color: #ff0000;'>Model Performance - Confusion Matrices</h3>", unsafe_allow_html=True)

def prepare_model_data(df):
    features = ['StudyTimeWeekly', 'Absences', 'ParentalSupport', 'Tutoring', 
                'Extracurricular', 'Sports', 'Music', 'Volunteering']
    
    median_gpa = df['GPA'].median()
    df['HighGPA'] = (df['GPA'] > median_gpa).astype(int)
    
    X = df[features].astype(float)
    y = df['HighGPA']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if len(filtered_df) > 10:
    try:
        X_train, X_test, y_train, y_test = prepare_model_data(filtered_df)

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        cm_lr = confusion_matrix(y_test, y_pred_lr)

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        col_cm1, col_cm2 = st.columns(2)

        with col_cm1:
            st.markdown("<h4 style='color: #ff0000; text-align: center;'>Logistic Regression</h4>", unsafe_allow_html=True)
            
            fig_cm_lr = go.Figure(data=go.Heatmap(
                z=cm_lr,
                x=['Predicted Low GPA', 'Predicted High GPA'],
                y=['Actual Low GPA', 'Actual High GPA'],
                colorscale='Reds',
                showscale=True
            ))
            
            fig_cm_lr.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#ff0000',
                xaxis_title="Predicted",
                yaxis_title="Actual",
                xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
                yaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
                title=dict(
                    text='logistic Regression Confusion Matrix',
                    font=dict(color='#ff0000', size=12),
                    x=0.5,
                    y=0.95
                ),
                height=500,
                width=None,
                autosize=True,
                margin=dict(l=80, r=80, t=80, b=80)
            )
            
            st.plotly_chart(fig_cm_lr, use_container_width=True, config={'displayModeBar': False})
            
            accuracy = accuracy_score(y_test, y_pred_lr)
            precision = precision_score(y_test, y_pred_lr)
            recall = recall_score(y_test, y_pred_lr)
            f1 = f1_score(y_test, y_pred_lr)
            
            st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <p>Accuracy: {accuracy:.2f}</p>
                <p>Precision: {precision:.2f}</p>
                <p>Recall: {recall:.2f}</p>
                <p>F1 Score: {f1:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_cm2:
            st.markdown("<h4 style='color: #ff0000; text-align: center;'>Random Forest</h4>", unsafe_allow_html=True)
            
            fig_cm_rf = go.Figure(data=go.Heatmap(
                z=cm_rf,
                x=['Predicted Low GPA', 'Predicted High GPA'],
                y=['Actual Low GPA', 'Actual High GPA'],
                colorscale='Reds',
                showscale=True
            ))
            
            fig_cm_rf.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#ff0000',
                xaxis_title="Predicted",
                yaxis_title="Actual",
                xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
                yaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
                title=dict(
                    text='Random Forest Confusion Matrix',
                    font=dict(color='#ff0000', size=12),
                    x=0.5,
                    y=0.95
                ),
                height=500,
                width=None,
                autosize=True,
                margin=dict(l=80, r=80, t=80, b=80)
            )
            
            st.plotly_chart(fig_cm_rf, use_container_width=True, config={'displayModeBar': False})
            
            accuracy = accuracy_score(y_test, y_pred_rf)
            precision = precision_score(y_test, y_pred_rf)
            recall = recall_score(y_test, y_pred_rf)
            f1 = f1_score(y_test, y_pred_rf)
            
            st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <p>Accuracy: {accuracy:.2f}</p>
                <p>Precision: {precision:.2f}</p>
                <p>Recall: {recall:.2f}</p>
                <p>F1 Score: {f1:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background-color: rgba(0, 0, 0, 0.7); border: 1px solid #ff0000; border-radius: 10px; padding: 20px; margin: 20px;'>
            <h4 style='color: #ff0000;'>Understanding Confusion Matrices</h4>
            <p>These confusion matrices show how well our models predict whether a student will have a high GPA (above median) or low GPA (below median).</p>
            <p><strong>True Positives:</strong> Students correctly predicted to have high GPA</p>
            <p><strong>True Negatives:</strong> Students correctly predicted to have low GPA</p>
            <p><strong>False Positives:</strong> Students incorrectly predicted to have high GPA</p>
            <p><strong>False Negatives:</strong> Students incorrectly predicted to have low GPA</p>
            <p>The models use features like study time, absences, parental support, tutoring, and extracurricular activities to make predictions.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='color: #ff0000;'>Feature Importance</h3>", unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'Feature': ['Study Time', 'Absences', 'Parental Support', 'Tutoring', 
                       'Extracurricular', 'Sports', 'Music', 'Volunteering'],
            'Importance': rf_model.feature_importances_
        })
        
        feature_importance = feature_importance.sort_values('Importance', ascending=True)
        
        fig_importance = go.Figure(go.Bar(
            x=feature_importance['Importance'],
            y=feature_importance['Feature'],
            orientation='h',
            marker_color='#ff0000'
        ))
        
        fig_importance.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text='Feature Importance in GPA Prediction',
                font=dict(color='#ff0000', size=20),
                x=0.5,
                y=0.95
            ),
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            xaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
            yaxis=dict(gridcolor='rgba(255,0,0,0.2)'),
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("""
        <div style='background-color: rgba(0, 0, 0, 0.7); border: 1px solid #ff0000; border-radius: 10px; padding: 20px; margin: 20px;'>
            <h4 style='color: #ff0000;'>Understanding Feature Importance</h4>
            <p>This chart shows how much each factor contributes to predicting a student's GPA:</p>
            <p><strong>Higher bars</strong> indicate factors that have a stronger influence on GPA prediction</p>
            <p><strong>Lower bars</strong> indicate factors that have less influence on GPA prediction</p>
            <p>The importance scores are calculated based on how often each feature is used to split the data in the Random Forest model.</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
else:
    st.warning("Not enough data points to generate confusion matrices. Please adjust the filters to include more data.")
