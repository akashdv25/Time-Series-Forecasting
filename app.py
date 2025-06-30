import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="System Monitoring & Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Hide sidebar completely */
    .css-1d391kg {display: none;}
    .css-1cypcdb {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .forecast-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3, .forecast-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.9;
        line-height: 1.2;
    }
    .metric-card h2, .forecast-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.historical_data = None
        self.cpu_forecast = None
        self.memory_forecast = None
        self.load_data()
    
    def load_data(self):
        """Load historical data and forecasts"""
        try:
            # Load historical data
            if os.path.exists("data/preprocessed/system_metrics_preprocessed.csv"):
                self.historical_data = pd.read_csv(
                    "data/preprocessed/system_metrics_preprocessed.csv",
                    index_col='timestamp',
                    parse_dates=True
                )
                # Ensure timezone consistency
                if self.historical_data.index.tz is None:
                    self.historical_data.index = self.historical_data.index.tz_localize('UTC')
                else:
                    self.historical_data.index = self.historical_data.index.tz_convert('UTC')
            
            # Load forecasts
            self.load_forecasts()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.historical_data = None
    
    def load_forecasts(self):
        """Load forecast data from CSV files"""
        try:
            if os.path.exists("artifacts/cpu_usage_percent_forecast.csv"):
                cpu_df = pd.read_csv("artifacts/cpu_usage_percent_forecast.csv")
                cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'], utc=True)
                self.cpu_forecast = cpu_df.set_index('timestamp')
            
            if os.path.exists("artifacts/memory_usage_percent_forecast.csv"):
                memory_df = pd.read_csv("artifacts/memory_usage_percent_forecast.csv")
                memory_df['timestamp'] = pd.to_datetime(memory_df['timestamp'], utc=True)
                self.memory_forecast = memory_df.set_index('timestamp')
                
        except Exception as e:
            st.warning(f"Could not load forecasts: {e}")
            self.cpu_forecast = None
            self.memory_forecast = None
    
    def get_model_info(self):
        """Get information about trained models"""
        model_info = {}
        artifacts_dir = "artifacts"
        
        if os.path.exists(artifacts_dir):
            for model_type in ['cpu_usage_percent', 'memory_usage_percent']:
                model_file = f"{artifacts_dir}/{model_type}_arima_model.pkl"
                if os.path.exists(model_file):
                    mod_time = os.path.getmtime(model_file)
                    model_info[model_type] = datetime.fromtimestamp(mod_time)
        
        return model_info
    
    def create_combined_chart(self, metric_name, historical_col, forecast_data, color_hist, color_forecast):
        """Create a combined historical + forecast chart"""
        try:
            fig = go.Figure()
            
            if self.historical_data is not None:
                # Historical data
                fig.add_trace(go.Scatter(
                    x=self.historical_data.index,
                    y=self.historical_data[historical_col],
                    mode='lines+markers',
                    name=f'Historical {metric_name}',
                    line=dict(color=color_hist, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>Historical {metric_name}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Value: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ))
            
            if forecast_data is not None:
                # Forecast data
                forecast_col = f"{historical_col}_forecast"
                if forecast_col in forecast_data.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data[forecast_col],
                        mode='lines+markers',
                        name=f'Forecasted {metric_name}',
                        line=dict(color=color_forecast, width=3, dash='dash'),
                        marker=dict(size=6, symbol='diamond'),
                        hovertemplate=f'<b>Forecasted {metric_name}</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Value: %{y:.2f}%<br>' +
                                     '<extra></extra>'
                    ))
                    
                    # Note: Removed forecast start annotation to avoid visual clutter
            
            # Update layout
            fig.update_layout(
                title=f"{metric_name} - Historical vs Forecasted",
                xaxis_title="Time",
                yaxis_title=f"{metric_name} (%)",
                hovermode='x unified',
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure if there's an error
            fig = go.Figure()
            fig.update_layout(
                title=f"{metric_name} - Error Loading Chart",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle',
                        showarrow=False, font=dict(size=16)
                    )
                ]
            )
            return fig
    
    def create_forecast_only_chart(self):
        """Create a chart showing only forecasts for both metrics"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('CPU Usage Forecast', 'Memory Usage Forecast'),
                vertical_spacing=0.1
            )
            
            if self.cpu_forecast is not None and 'cpu_usage_percent_forecast' in self.cpu_forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.cpu_forecast.index,
                        y=self.cpu_forecast['cpu_usage_percent_forecast'],
                        mode='lines+markers',
                        name='CPU Forecast',
                        line=dict(color='#ff6b6b', width=3),
                        marker=dict(size=8, symbol='circle')
                    ),
                    row=1, col=1
                )
            
            if self.memory_forecast is not None and 'memory_usage_percent_forecast' in self.memory_forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.memory_forecast.index,
                        y=self.memory_forecast['memory_usage_percent_forecast'],
                        mode='lines+markers',
                        name='Memory Forecast',
                        line=dict(color='#4ecdc4', width=3),
                        marker=dict(size=8, symbol='square')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title="Next 4 Hours Forecast",
                height=600,
                template='plotly_white',
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="CPU Usage (%)", row=1, col=1)
            fig.update_yaxes(title_text="Memory Usage (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            # Return empty figure if there's an error
            fig = go.Figure()
            fig.update_layout(
                title="Forecast Chart - Error Loading",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle',
                        showarrow=False, font=dict(size=16)
                    )
                ]
            )
            return fig
    
    def display_metrics(self):
        """Display current metrics and model info"""
        col1, col2, col3, col4 = st.columns(4)
        
        model_info = self.get_model_info()
        
        with col1:
            if self.historical_data is not None:
                latest_cpu = self.historical_data['cpu_usage_percent'].iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current CPU USAGE</h3>
                    <h2>{latest_cpu:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Current CPU Usage", "No Data", "")
        
        with col2:
            if self.historical_data is not None:
                latest_memory = self.historical_data['memory_usage_percent'].iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Memory Usage</h3>
                    <h2>{latest_memory:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Current Memory Usage", "No Data", "")
        
        with col3:
            try:
                if (self.cpu_forecast is not None and 
                    'cpu_usage_percent_forecast' in self.cpu_forecast.columns and 
                    len(self.cpu_forecast) > 0):
                    avg_cpu = self.cpu_forecast['cpu_usage_percent_forecast'].mean()
                    st.markdown(f"""
                    <div class="forecast-card">
                        <h3>Forecasted CPU Usage (4hr avg)</h3>
                        <h2>{avg_cpu:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("Next CPU", "No Forecast", "")
            except Exception:
                st.metric("Next CPU", "Error", "")
        
        with col4:
            try:
                if (self.memory_forecast is not None and 
                    'memory_usage_percent_forecast' in self.memory_forecast.columns and 
                    len(self.memory_forecast) > 0):
                    avg_memory = self.memory_forecast['memory_usage_percent_forecast'].mean()
                    st.markdown(f"""
                    <div class="forecast-card">
                        <h3>Forecasted Memory Usage(4hr avg)</h3>
                        <h2>{avg_memory:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("Next Memory", "No Forecast", "")
            except Exception:
                st.metric("Next Memory", "Error", "")
    

    
    def run(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">📊 System Monitoring & Forecasting Dashboard</h1>', unsafe_allow_html=True)
        
        # Check if models exist
        model_info = self.get_model_info()
        models_exist = len(model_info) > 0
        
        # Check if historical data exists
        if self.historical_data is None:
            st.error("❌ **No historical data available**")
            st.markdown("""
            ### 🚀 **Getting Started**
            
            To use this dashboard, you need to run the complete ML pipeline first:
            
            ```bash
            # Run the complete pipeline
            python src/main.py
            ```
                        
              ```bash
            # Restart the streamlit container
            docker compose restart streamlit
            ```             
            
            This will:
            1. 📊 Collect system metrics from InfluxDB
            2. 🧹 Clean and preprocess the data  
            3. 🤖 Train ARIMA forecasting models
            4. 🔮 Generate forecasts for the next 4 hours
            
            Once completed, refresh this page to see your forecasting dashboard!
            """)
            return
        
        # Check if models exist
        if not models_exist:
            st.warning("⚠️ **No trained models found**")
            st.markdown("""
            ### 🤖 **Models Not Ready**
            
            Historical data is available, but no forecasting models have been trained yet.
            
            ```bash
            # Train the models
            python src/model_train.py
            ```
            
            Or run the complete pipeline:
            ```bash
            python src/main.py
            ```
            
            After training completes, refresh this page to see forecasts!
            """)
            
            # Show historical data only
            st.markdown("---")
            st.subheader("📊 Historical Data Preview")
            
            # Show basic metrics without forecasts
            col1, col2 = st.columns(2)
            with col1:
                latest_cpu = self.historical_data['cpu_usage_percent'].iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current CPU Usage</h3>
                    <h2>{latest_cpu:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                latest_memory = self.historical_data['memory_usage_percent'].iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Memory Usage</h3>
                    <h2>{latest_memory:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Show historical charts only
            tab1, tab2 = st.tabs(["📈 CPU History", "💾 Memory History"])
            
            with tab1:
                cpu_chart = self.create_combined_chart(
                    "CPU Usage", 
                    "cpu_usage_percent", 
                    None,  # No forecast data
                    "#ff6b6b",
                    "#ff9999"
                )
                st.plotly_chart(cpu_chart, use_container_width=True)
            
            with tab2:
                memory_chart = self.create_combined_chart(
                    "Memory Usage", 
                    "memory_usage_percent", 
                    None,  # No forecast data
                    "#4ecdc4",
                    "#7fdddd"
                )
                st.plotly_chart(memory_chart, use_container_width=True)
            
            return
        
        # Full dashboard with forecasts
        # Metrics row
        self.display_metrics()
        
        st.markdown("---")
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["📈 CPU Analysis", "💾 Memory Analysis", "🔮 Forecasts Only"])
        
        with tab1:
            st.subheader("CPU Usage Analysis")
            cpu_chart = self.create_combined_chart(
                "CPU Usage", 
                "cpu_usage_percent", 
                self.cpu_forecast,
                "#ff6b6b",  # Red for historical
                "#ff9999"   # Light red for forecast
            )
            st.plotly_chart(cpu_chart, use_container_width=True)
        
        with tab2:
            st.subheader("Memory Usage Analysis")
            memory_chart = self.create_combined_chart(
                "Memory Usage", 
                "memory_usage_percent", 
                self.memory_forecast,
                "#4ecdc4",  # Teal for historical
                "#7fdddd"   # Light teal for forecast
            )
            st.plotly_chart(memory_chart, use_container_width=True)
        
        with tab3:
            st.subheader("Forecast Comparison")
            forecast_chart = self.create_forecast_only_chart()
            st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("**🤖 Powered by SARIMA Models | 📊 Real-time System Monitoring**")

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run()