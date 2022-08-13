# Feladat részletezése
## Feladat megoldásának lépései:
-	Napi adatok letöltése 
-	napi, heti, havi, és negyedéves `maximum high`, `minimum low`, értékek kiszámítása és DataFramehez csatolása
-	Unit teszt 

```Python
#A dátum oszlopot Index 
df = df.set_index(pd.to_datetime(df['Date']))
```

```Python
#A dátum oszlopot Index 
df = df.set_index(pd.to_datetime(df['Date']))
```

 ![Az összes meccs adatát tartalmazó dataframe](pics/Date_index.png)

 Létrehoztam egy previous_week oszlopot, ami alapján kiszámítottam a heti High maximumokat illetve heti Low minimumokat

 ```Python
#Create new (previous_week) column 
# Based on the previous week column group data in a weekly basis and calculate the maximum High/minimum Low values

df['Prev_Week_date'] = df['Date']-pd.to_timedelta(7, unit='d')
df['Weekly_Max_high'] = df.groupby([pd.Grouper(key = 'Prev_Week_date',freq = 'W')])['High'].transform('max')
df['Weekly_Min_low'] = df.groupby([pd.Grouper(key = 'Prev_Week_date',freq = 'W')])['Low'].transform('min')
```

Létrehoztam egy Quarter illetve egy Prev_Quarter oszlopot, majd a Prev_Quarter oszlop alapján kiszámítottam a maximum High és minimum Low értékeket.
Miután előálltak az új oszlopok, hozzáillesztettem az előző DataFramhez.

```Python
#Create new (Quarter and Prev_Quarter) columns 

df['Quarter'] = df['Date'].dt.to_period('Q')
df['Prev_Quarter'] = df['Date'].dt.to_period('Q') - 1

# Group data in a quarterly basis and calculate the maximum High values

Quarterly_Max_high_df = df.groupby('Quarter')['High'].max().reset_index()
Quarterly_Max_high_df.rename(columns = {'High':'Quarterly_max_High','Quarter':'Prev_Quarter',}, inplace = True)
Quarterly_Max_high_df

# Group data in a quarterly basis and calculate the minimum Low values

Quarterly_Min_low_df = df.groupby('Quarter')['Low'].min().reset_index()
Quarterly_Min_low_df.rename(columns = {'Low':'Quarterly_min_Low','Quarter':'Prev_Quarter',}, inplace = True)

#Join the results to the previous DataFrame 

Merged_df = df.merge(Quarterly_Max_high_df, how='left', on=["Prev_Quarter"])
Merged_df = Merged_df.merge(Quarterly_Min_low_df, how='left', on=["Prev_Quarter"])
```

Létrehoztam egy Month illetve egy Prev_MOnth oszlopot, majd a Prev_MOnth oszlop alapján kiszámítottam a maximum High és minimum Low értékeket.
Miután előálltak az új oszlopok, hozzáillesztettem az előző DataFramhez.

```Python
#Create new (Month and Prev_Month) columns 

Merged_df['Month'] = Merged_df['Date'].dt.to_period('M')
Merged_df['Prev_Month'] = Merged_df['Date'].dt.to_period('M') - 1

# Group data in monthly basis and calculate the maximum High values

Monthly_Max_high_df = Merged_df.groupby('Month')['High'].max().reset_index()
Monthly_Max_high_df.rename(columns = {'High':'Monthly_max_High','Month':'Prev_Month',}, inplace = True)

# Group data in monthly basis and calculate the minimum Low values

Monthly_Min_low_df = Merged_df.groupby('Month')['Low'].min().reset_index()
Monthly_Min_low_df.rename(columns = {'Low':'Monthly_min_low','Month':'Prev_Month',}, inplace = True)

#Join the results to the previous DataFrame 

Merged_df_v2 = Merged_df.merge(Monthly_Max_high_df, how='left', on=["Prev_Month"])
Merged_df_v2 = Merged_df_v2.merge(Monthly_Min_low_df, how='left', on=["Prev_Month"])
```