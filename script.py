import pandas as pd

weather = pd.read_csv('laporan_iklim_harian.csv', index_col = "Tanggal")

output = weather.apply(pd.isnull).sum()/weather.shape[0]

core_weather = weather[["Tavg", "RH_avg","RR", "ss", "ff_avg"]].copy()

core_weather.columns = ["Temp_rata-rata", "Kelembapan_rata-rata","Curah-hujan","Lamanya-penyinaran-matahari","durasi_hujan","kelembapan_relatif"]



print(output)