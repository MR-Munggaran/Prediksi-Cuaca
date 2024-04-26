import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Inisialisasi aplikasi Dash
app = dash.Dash(_name_)

# Tampilan layout dashboard
app.layout = html.Div([
    html.H1("Dashboard Interaktif Prediksi Cuaca"),
    
    # Dropdown menu untuk memilih parameter
    dcc.Dropdown(
        id='parameter-dropdown',
        options=[
            {'label': 'Prediksi Suhu', 'value': 'suhu'},
            {'label': 'Prediksi Kelembapan', 'value': 'kelembapan'},
            {'label': 'Prediksi Curah Hujan', 'value': 'curah_hujan'},
            {'label': 'Prediksi Lamanya Penyinaran Matahari', 'value': 'lamanya_penyinaran'},
            {'label': 'Prediksi Durasi Hujan', 'value': 'durasi_hujan'},
        ],
        value='suhu',  # Nilai default
        style={'width': '50%'}
    ),
    
    # Grafik hasil prediksi
    dcc.Graph(id='prediksi-chart'),

    # Informasi MSE dan RMSE
    html.Div([
        html.Label(id='mse-label'),
        html.Label(id='rmse-label'),
    ])
])

@app.callback(
    [Output('prediksi-chart', 'figure'),
     Output('mse-label', 'children'),
     Output('rmse-label', 'children')],
    [Input('parameter-dropdown', 'value')]
)
def update_dashboard(parameter):
    # Membuat nama kolom yang sesuai dengan format yang dihasilkan dari label dropdown
    column_name = f'Prediksi {parameter.capitalize().replace(" ", "_")}'
    
    if parameter == 'suhu':
        hasil_df = pd.DataFrame({column_name: prediksi_suhu_skala_asli, 'Data Sebenarnya': y1_test_skala_asli})
        mse = mse1
        rmse = rmse1
    elif parameter == 'kelembapan':
        hasil_df = pd.DataFrame({column_name: prediksi_kelembapan_skala_asli, 'Data Sebenarnya': y2_test_skala_asli})
        mse = mse2
        rmse = rmse2
    elif parameter == 'curah_hujan':
        hasil_df = pd.DataFrame({column_name: prediksi_curah_hujan_skala_asli, 'Data Sebenarnya': y3_test_skala_asli})
        mse = mse3
        rmse = rmse3
    elif parameter == 'lamanya_penyinaran':
        hasil_df = pd.DataFrame({column_name: prediksi_lamanya_penyinaran_matahari_skala_asli, 'Data Sebenarnya': y4_test_skala_asli})
        mse = mse4
        rmse = rmse4
    elif parameter == 'durasi_hujan':
        hasil_df = pd.DataFrame({column_name: prediksi_durasi_hujan_skala_asli, 'Data Sebenarnya': y5_test_skala_asli})
        mse = mse5
        rmse = rmse5

    figure = {
        'data': [
            {'x': hasil_df.index, 'y': hasil_df['Data Sebenarnya'], 'type': 'bar', 'name': 'Data Sebenarnya'},
            {'x': hasil_df.index, 'y': hasil_df[column_name], 'type': 'bar', 'name': column_name},
        ],
        'layout': {
            'title': f'Perbandingan Data Sebenarnya dan {column_name}'
        }
    }

    return figure, f'Mean Squared Error: {mse:.4f}', f'Root Mean Squared Error: {rmse:.4f}'


if _name_ == '_main_':
    # Jalankan server pada localhost:8501
    app.run_server(debug=True)