import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

weather = pd.read_csv('Data_fixed.csv', index_col = "Tanggal")

output = weather.apply(pd.isnull).sum()/weather.shape[0]

core_weather = weather[["Tavg", "RH_avg","RR", "ss", "ff_avg"]].copy()

core_weather.columns = ["Temp_rata-rata", "Kelembapan_rata-rata","Curah-hujan","Lamanya-penyinaran-matahari","durasi_hujan"]
core_weather = core_weather.dropna()  # Untuk menghapus baris yang mengandung nilai-nilai hilang
# atau
core_weather = core_weather.fillna(core_weather.mean()) 

for col in core_weather.columns:
    min = core_weather[col].min()
    max = core_weather[col].max()
    core_weather[col +'-scaled'] = (core_weather[col] - min) / ( max - min )

x_uji1 = sm.add_constant(core_weather[["Kelembapan_rata-rata-scaled","Curah-hujan-scaled","Lamanya-penyinaran-matahari-scaled","durasi_hujan-scaled"]])
y_uji1 = core_weather["Temp_rata-rata-scaled"]

model1 = sm.OLS(y_uji1,x_uji1).fit()

def membuat_model_prediksi_suhu(a, b, c, d):
    """
    parameter a adalah parameter untuk nilai kelembapan
    parameter b adalah parametet untuk nilai curah hujan
    parameter c adalah parameter untuk lamanya penyinaran matahari
    parameter d adalah parameter untuk durasi hujan
    constant adalah nilai  ketika dari seluruh parameter bernilai 0
    """
    coef_kelembapan = -0.4089
    coef_curah_hujan = -0.0603
    coef_lamanya_penyinaran_matahari = 0.1233
    coef_durasi_hujan = -0.0700
    constant = 0.6894

    return a*coef_kelembapan + b * coef_curah_hujan + c * coef_lamanya_penyinaran_matahari + d * coef_durasi_hujan + constant
x_uji2 = sm.add_constant(core_weather[["Temp_rata-rata-scaled","Curah-hujan-scaled","Lamanya-penyinaran-matahari-scaled","durasi_hujan-scaled"]])
y_uji2 = core_weather["Kelembapan_rata-rata-scaled"]

model2 = sm.OLS(y_uji2,x_uji2).fit()


def membuat_model_prediksi_kelembapan(a, b, c, d):
    """
    parameter a adalah parameter untuk nilai suhu
    parameter b adalah parametet untuk nilai curah hujan
    parameter c adalah parameter untuk lamanya penyinaran matahari
    parameter d adalah parameter untuk durasi hujan
    constant adalah nilai  ketika dari seluruh parameter bernilai 0
    """
    coef_suhu = -0.4922
    coef_curah_hujan = -0.0429
    coef_lamanya_penyinaran_matahari = -0.3283
    coef_durasi_hujan = 0.0120
    constant = 0.9066

    return a*coef_suhu + b * coef_curah_hujan + c * coef_lamanya_penyinaran_matahari + d * coef_durasi_hujan + constant


x_uji3 = sm.add_constant(core_weather[["Temp_rata-rata-scaled","Kelembapan_rata-rata-scaled","Lamanya-penyinaran-matahari-scaled","durasi_hujan-scaled"]])
y_uji3 = core_weather["Curah-hujan-scaled"]

model3 = sm.OLS(y_uji3,x_uji3).fit()

def membuat_model_prediksi_curah_hujan(a, b, c, d):
    """
    parameter a adalah parameter untuk nilai suhu
    parameter b adalah parametet untuk nilai kelembapan
    parameter c adalah parameter untuk lamanya penyinaran matahari
    parameter d adalah parameter untuk durasi hujan
    constant adalah nilai  ketika dari seluruh parameter bernilai 0
    """
    coef_suhu = -0.5121
    coef_kelembapan = -0.3028
    coef_lamanya_penyinaran_matahari = 0.2075
    coef_durasi_hujan = -0.2186
    constant = 0.5509

    return a*coef_suhu + b * coef_kelembapan + c * coef_lamanya_penyinaran_matahari + d * coef_durasi_hujan + constant

x_uji4 = sm.add_constant(core_weather[["Temp_rata-rata-scaled","Kelembapan_rata-rata-scaled","Curah-hujan-scaled","durasi_hujan-scaled"]])
y_uji4 = core_weather["Lamanya-penyinaran-matahari-scaled"]

model4 = sm.OLS(y_uji4,x_uji4).fit()

def membuat_model_prediksi_curah_hujan(a, b, c, d):
    """
    parameter a adalah parameter untuk nilai suhu
    parameter b adalah parametet untuk nilai kelembapan
    parameter c adalah parameter untuk curah hujan
    parameter d adalah parameter untuk durasi hujan
    constant adalah nilai  ketika dari seluruh parameter bernilai 0
    """
    coef_suhu = 0.3409
    coef_kelembapan = -0.7537
    coef_curah_hujan = 0.0675
    coef_durasi_hujan = -0.0258
    constant = 0.6169

    return a*coef_suhu + b * coef_kelembapan + c * coef_curah_hujan + d * coef_durasi_hujan + constant


x_uji5 = sm.add_constant(core_weather[["Temp_rata-rata-scaled","Kelembapan_rata-rata-scaled","Curah-hujan-scaled","Lamanya-penyinaran-matahari-scaled"]])
y_uji5 = core_weather["durasi_hujan-scaled"]

model5 = sm.OLS(y_uji5,x_uji5).fit()

def membuat_model_prediksi_curah_hujan(a, b, c, d):
    """
    parameter a adalah parameter untuk nilai suhu
    parameter b adalah parametet untuk nilai kelembapan
    parameter c adalah parameter untuk curah hujan
    parameter d adalah parameter untuk nilai lamanya penyinaran matahari
    constant adalah nilai  ketika dari seluruh parameter bernilai 0
    """
    coef_suhu = -0.4426
    coef_kelembapan = 0.0632
    coef_curah_hujan = -0.1626
    coef_lamanya_penyinaran_matahari = -0.0590
    constant = 0.5514

    return a*coef_suhu + b * coef_kelembapan + c * coef_curah_hujan + d * coef_lamanya_penyinaran_matahari + constant

# split data dengan sklearn train test split
# split data untuk prediksi suhu
x1 = x_uji1
y1 = y_uji1
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

# split data untuk prediksi kelembapan
x2 = x_uji2
y2 = y_uji2
X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)

# split data untuk prediksi curah hujan
x3 = x_uji3
y3 = y_uji3
X3_train, X3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)

# split data untuk prediksi lamanya penyinaran matahari

x4 = x_uji4
y4 = y_uji4
X4_train, X4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.2, random_state=42)

# split data untuk prediksi durasi hujan

x5 = x_uji5
y5 = y_uji5
X5_train, X5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.2, random_state=42)


prediksi = LinearRegression()

prediksi.fit(X1_train, y1_train)

prediksi_suhu = prediksi.predict(X1_test)

prediksi.fit(X2_train, y2_train)
prediksi_kelembapan = prediksi.predict(X2_test)

prediksi.fit(X3_train, y3_train)
prediksi_curah_hujan = prediksi.predict(X3_test)

prediksi.fit(X4_train, y4_train)
prediksi_lamanya_penyinaran_matahari = prediksi.predict(X4_test)

prediksi.fit(X5_train, y5_train)
prediksi_durasi_hujan = prediksi.predict(X5_test)

# Ambil kolom suhu dari core_weather
suhu_column = core_weather['Temp_rata-rata']

# Hitung nilai minimum dan maksimum dari kolom suhu
min_suhu = suhu_column.min()
max_suhu = suhu_column.max()

# Buat DataFrame untuk menyimpan hasil inversi scaling
prediksi_suhu_baru = pd.DataFrame()

# Lakukan inversi scaling hanya untuk kolom suhu
# for col in prediksi_suhu:
#     prediksi_suhu_baru[col] = (prediksi_suhu[col]) * (max_suhu - min_suhu) + min_suhu

prediksi_suhu_skala_asli = (prediksi_suhu * (max_suhu - min_suhu)) + min_suhu


# Ambil kolom suhu dari core_weather
kelembapan = core_weather['Kelembapan_rata-rata']

# Hitung nilai minimum dan maksimum dari kolom suhu
min_kelembapan = kelembapan.min()
max_kelembapan = kelembapan.max()

# Buat DataFrame untuk menyimpan hasil inversi scaling

# Lakukan inversi scaling hanya untuk kolom suhu
# for col in prediksi_suhu:
#     prediksi_suhu_baru[col] = (prediksi_suhu[col]) * (max_suhu - min_suhu) + min_suhu

prediksi_kelembapan_skala_asli = (prediksi_kelembapan * (max_kelembapan - min_kelembapan)) + min_kelembapan

# Ambil kolom suhu dari core_weather
curah_hujan = core_weather['Curah-hujan']

# Hitung nilai minimum dan maksimum dari kolom suhu
min_curah_hujan = curah_hujan.min()
max_curah_hujan = curah_hujan.max()


prediksi_curah_hujan_skala_asli = (prediksi_curah_hujan * (max_curah_hujan - min_curah_hujan)) + min_curah_hujan

# Ambil kolom suhu dari core_weather
lamanya_penyinaran_matahari = core_weather['Lamanya-penyinaran-matahari']

# Hitung nilai minimum dan maksimum dari kolom suhu
min_lamanya_penyinaran_matahari = lamanya_penyinaran_matahari.min()
max_lamanya_penyinaran_matahari = lamanya_penyinaran_matahari.max()


prediksi_lamanya_penyinaran_matahari_skala_asli = (prediksi_lamanya_penyinaran_matahari * (max_lamanya_penyinaran_matahari - min_lamanya_penyinaran_matahari)) + min_lamanya_penyinaran_matahari


# Ambil kolom suhu dari core_weather
durasi_hujan = core_weather['durasi_hujan']

# Hitung nilai minimum dan maksimum dari kolom suhu
min_durasi_hujan = durasi_hujan.min()
max_durasi_hujan = durasi_hujan.max()


prediksi_durasi_hujan_skala_asli = (prediksi_durasi_hujan * (max_durasi_hujan - min_durasi_hujan)) + min_durasi_hujan

y1_test_skala_asli = (y1_test * (max_suhu - min_suhu)) + min_suhu


mse1 = mean_squared_error(y1_test_skala_asli, prediksi_suhu_skala_asli)
rmse1 = np.sqrt(mse1)

y2_test_skala_asli = (y2_test * (max_kelembapan - min_kelembapan)) + min_kelembapan


mse2 = mean_squared_error(y2_test_skala_asli, prediksi_kelembapan_skala_asli)
rmse2 = np.sqrt(mse2)

y3_test_skala_asli = (y3_test * (max_curah_hujan - min_curah_hujan)) + min_curah_hujan


mse3 = mean_squared_error(y3_test_skala_asli, prediksi_curah_hujan_skala_asli)
rmse3 = np.sqrt(mse3)

y4_test_skala_asli = (y4_test * (max_lamanya_penyinaran_matahari - min_lamanya_penyinaran_matahari)) + min_lamanya_penyinaran_matahari


mse4 = mean_squared_error(y4_test_skala_asli, prediksi_lamanya_penyinaran_matahari_skala_asli)
rmse4 = np.sqrt(mse4)

y5_test_skala_asli = (y5_test * (max_durasi_hujan - min_durasi_hujan)) + min_durasi_hujan


mse5 = mean_squared_error(y5_test_skala_asli, prediksi_lamanya_penyinaran_matahari_skala_asli)
rmse5 = np.sqrt(mse5)

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