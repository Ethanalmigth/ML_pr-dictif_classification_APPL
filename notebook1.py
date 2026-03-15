# %%
from pyexpat import features

print("Ca commence")
# %%
import yfinance as YF
import numpy as np
import pandas as pd
# %%
try:
    data = YF.download("AAPL", start="2020-01-01", end="2023-01-01")
    if data.empty:
        raise ValueError("No data")
    data.to_csv('AAPL.csv')
    print("✅ Données téléchargées et sauvegardées")
except Exception as e:
    print(f"Une erreur est survenue {e}")
    data = pd.read_csv("AAPL.csv")
    print("📂 Données chargées depuis le CSV")

# %%
data.columns= data.columns.get_level_values(0)
data=data.dropna()
# %%
data["Return"] = data["Close"].pct_change()

# %%
data
# %%
data["MA_20"]= data["Close"].rolling(20).mean()
data
# %%
data["MA_50"]= data["Close"].rolling(50).mean()
# %%
data["MA_200"]=data["Close"].rolling(200).mean()
# %%
data["Volatility"]= data["Return"].rolling(20).std()
# %% [markdown]
# momentum
# MA ratio
# RSI
# volume change
# distance to MA
# 
# %%
data["Momentum"]=data["Close"].pct_change(periods=10)
data
# %%
data["MA_ratio"]=data["MA_20"]/data["MA_50"]
# %%
data
# %%
def calculate_rsi(data, period=14):
    # Calculer les variations de prix
    delta = data['Close'].diff()

    # Séparer gains et pertes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculer les moyennes mobiles
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculer RS et RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Appliquer la fonction
data["RSI"] = calculate_rsi(data, period=14)

# %%
data
# %%
data["Volume_Change"] = data["Volume"].pct_change()
data
# %%
data["Distance_to_MA"]=((data["Close"] - data["MA_50"])/data["MA_50"])
# %%
data
# %%
data= data.dropna()
data
# %%
data.to_csv('AAPL_with_indicators.csv')
# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Création d'une figure avec des lignes de hauteurs différentes
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('AAPL - Prix & MAs', 'RSI', 'Distance à la MA 50'),
                    row_heights=[0.5, 0.25, 0.25])

# 1. Graphique du Prix (Candlestick)
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='MA 50'), row=1, col=1)

# 2. RSI
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

# 3. Distance à la MA (Indicateur d'étirement)
fig.add_trace(go.Bar(x=data.index, y=data['Distance_to_MA'], name='Dist MA 50'), row=3, col=1)

# Mise en page
fig.update_layout(height=800, title_text="Analyse Technique Interactive - AAPL", showlegend=True)

# %%
data["Target"]= (data["Close"]>data["Close"].shift(-7))
data
# %%
data["Target"]=data["Target"].astype(int)
# %%
data.
# %%
taille= int(len(data)*0.8)

train=data[:taille]
test=data[taille:]
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

accuracy = []
precision = []
recall = []
f1= []
features= ["Return","Volatility","Momentum","MA_ratio","Volume_Change","RSI","Distance_to_MA"]
ts=TimeSeriesSplit(n_splits=5)
X=data[features]
Y=data["Target"]

for i,j in ts.split(X):
    model = RandomForestClassifier(random_state=42)
    x_train=X.iloc[i]
    y_train=Y.iloc[i]
    x_test=X.iloc[j]
    y_test=Y.iloc[j]
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy.append(accuracy_score(y_test, prediction))
    precision.append(precision_score(y_test, prediction))
    recall.append(recall_score(y_test, prediction))
    f1.append(f1_score(y_test, prediction))

# %%
for i,j in enumerate(accuracy):
    print(f"le modele {i}  a {j} de accuracy")
# %%
for i,j in enumerate(precision):
    print(f"le modele {i}  a {j} de precision")
# %%
for i,j in enumerate(recall):
    print(f"le modele {i}  a {j} de recall")
# %%
for i,j in enumerate(f1):
    print(f"le modele {i}  a {j} de f1_score")
# %%
print(f"Accuracy moyenne : {np.mean(accuracy):.2f} ecart-type : {np.std(accuracy):.2f}")
print(f"F1-Score moyen : {np.mean(f1):.2f} ecart-type : {np.std(f1):.2f}")
print(f"Precision moyenne : {np.mean(precision):.2f} ecart-type : {np.std(precision):.2f}")
print(f"Recall moyenne : {np.mean(recall):.2f}  ecart-type : {np.std(recall):.2f} ")

# %% [markdown]
# Backtesting
# %% [markdown]
# Tableau des metrique du modele
# 
# %%
tableau_metrique=[]
for i in range(5):
    tab=[]
    tab.append(i+1)
    tab.append(accuracy[i])
    tab.append(precision[i])
    tab.append(recall[i])
    tab.append(f1[i])
    tableau_metrique.append(tab)

pd_metrique= pd.DataFrame(tableau_metrique, columns=["Modele","Accuracy","Precision","Recall","F1"])
pd_metrique
# %%
data_test= data.copy()

# %%
