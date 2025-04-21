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
