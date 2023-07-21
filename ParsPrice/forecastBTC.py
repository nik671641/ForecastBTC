import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('bitcoin_history.csv')

# Преобразуем столбец с датами в объект datetime и отсортируем данные по дате
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Выбираем столбец с ценами биткойнов для прогнозирования
price_data = data[['Close']].values.astype(float)

# Нормализуем данные
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_price_data = scaler.fit_transform(price_data)

# Функция для создания обучающих данных
def create_dataset(data, look_back=1, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back:i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(y)

# Задаем параметры модели
look_back = 300  # число временных шагов для обучения (последние 300 дней)
forecast_horizon = 30  # число временных шагов для прогноза (30 дней)
X, y = create_dataset(normalized_price_data, look_back, forecast_horizon)

# Разделяем данные на обучающую и тестовую выборки
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Подготовка данных для LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Создание модели LSTM
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(forecast_horizon))

# Компилируем модель
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, forecast_horizon))

# Визуализация результатов с использованием plotly
fig = go.Figure()

# График истории цен за последний год
last_year_data = data[data['Date'] >= (data['Date'].max() - pd.DateOffset(years=1))]
fig.add_trace(go.Scatter(x=last_year_data['Date'], y=last_year_data['Close'], mode='lines', name='История цен за последний год'))

# График прогноза
forecast_dates = data['Date'].values[-forecast_horizon:]
fig.add_trace(go.Scatter(x=forecast_dates, y=y_pred.flatten(), mode='lines', name='Прогноз на 30 дней', line=dict(color='red')))

# Добавление возможности видеть цену при наведении мышкой
hover_template = '<b>Дата:</b> %{x}<br><b>Цена биткойна:</b> %{y:.2f}'
fig.update_traces(hovertemplate=hover_template)

# Настройка макета графика
fig.update_layout(title='История цен и прогноз на 30 дней',
                  xaxis_title='Дата',
                  yaxis_title='Цена биткойна',
                  xaxis=dict(type='category'),
                  template='plotly_white')

# Отображение интерактивного графика
fig.show()