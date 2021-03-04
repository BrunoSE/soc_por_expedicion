import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

carpeta = 'graficos_2020_11_02_2021_02_22'
dibujar_3d = True
dibujar_2d = True
archivo_data = f'data_Laboral_{carpeta}.parquet'

os.chdir(carpeta)
if not os.path.isdir('Anomalias_dsoc'):
    os.mkdir('Anomalias_dsoc')

os.chdir('Anomalias_dsoc')

if not os.path.isdir('graficos_2d'):
    os.mkdir('graficos_2d')


df = pd.read_parquet(archivo_data)
df = df[['distancia_recorrida', 'delta_soc', 'tiempo_viaje',
         'codigo_Ruta', 'MH_inicio', 'PPU', 'valor_soc_Ttec_ini',
         'Indice_mensual', 'delta_Pcon', 'delta_Pgen', 'V_Comercial']]

rutas = df['codigo_Ruta'].unique()


def mh_a_int(x):
    return int(int(x[:2]) * 2 + int(x[3:5]) / 30)


def texto_dsoc(xP, xS, xD, xT, xMH, xSi):
    return (f'{xP}<br>' + str(xMH)[:-3] + '<br>' + f'delta Soc: {xS:.2f}<br>' +
            f'distancia: {xD:.0f}<br>' + f'tiempo: {xT:.0f}<br>' + f'Soc Inicial: {xSi:.2f}')


def texto_pnet(xP, xS, xD, xT, xMH, xSi):
    return (f'{xP}<br>' + str(xMH)[:-3] + '<br>' + f'Potencia: {xS:.2f}<br>' +
            f'distancia: {xD:.0f}<br>' + f'tiempo: {xT:.0f}<br>' + f'Soc Inicial: {xSi:.2f}')


df['MH_inicio2'] = df.apply(lambda x: mh_a_int(x['MH_inicio']), axis=1)
# hacer diccionario media hora - numero de media hora
Diccionario_MH = df[['MH_inicio', 'MH_inicio2']].copy()
Diccionario_MH.drop_duplicates(inplace=True)
Diccionario_MH.sort_values(by='MH_inicio2', inplace=True)
Diccionario_MH['MH_inicio'] = Diccionario_MH['MH_inicio'].str[:-3]
Diccionario_MH.set_index('MH_inicio2', drop=True, inplace=True)
Diccionario_MH = Diccionario_MH.to_dict()['MH_inicio']


mhs = df['MH_inicio2'].unique()
mhs.sort()
df['distancia_recorrida'] = df['distancia_recorrida'] / 1000
df['delta_soc'] = df['delta_soc'] * 100
df['Potencia_neta'] = df['delta_Pcon'] - df['delta_Pgen']

cortes = [x for x in range(0, 48 + 1, 12)]

colores = ['#000000',
           '#E59400',
           '#989898',
           '#FF0000',
           "#0046c4",
           "#006f10",
           "#0d004a",
           "#8dac4b",
           "#7b0048",
           "#009c91",
           "#af94e2",
           "#513400"]


if dibujar_2d:

    dibujar_tv_dsoc = False
    dibujar_pnet_dsoc = False
    dibujar_dsoc_si = True

    sns.set_style("whitegrid")
    # sns.color_palette("husl", 8)
    cmap = ListedColormap(sns.color_palette("viridis", 256))

    plt.rcParams['figure.figsize'] = [16, 8]
    max_dsoc = df['delta_soc'].max() + 2
    max_tv = df['tiempo_viaje'].max() + 2
    max_pnet = df['Potencia_neta'].max() + 2
    max_si = df['valor_soc_Ttec_ini'].max() + 2

    min_dsoc = min(df['delta_soc'].min(), 0) - 2
    min_tv = min(df['tiempo_viaje'].min(), 0) - 2
    min_pnet = min(df['Potencia_neta'].min(), 0) - 2
    min_si = min(df['valor_soc_Ttec_ini'].min(), 0) - 2

    if dibujar_tv_dsoc:
        for ss in df['codigo_Ruta'].unique():
            print(f'Graficando 2d tv_dsoc {ss}')
            dfx_ = df.loc[df['codigo_Ruta'] == ss].copy()
            corrS = dfx_[['delta_soc', 'tiempo_viaje']].corr(method='spearman')
            corrP = dfx_[['delta_soc', 'tiempo_viaje']].corr(method='pearson')

            fig = plt.figure()
            grafico = sns.scatterplot(x="delta_soc",
                                      y="tiempo_viaje",
                                      hue='V_Comercial',
                                      palette=cmap,
                                      data=dfx_
                                      )

            fig = grafico.get_figure()
            plt.legend(title='V [km/h]')
            plt.xlim(min_dsoc, max_dsoc)
            plt.ylim(min_tv, max_tv)
            plt.annotate(f'Correlacion (S): {round(corrS.iloc[0][1], 3):.3f}\n'
                         f'Correlacion (P): {round(corrP.iloc[0][1], 3):.3f}',
                         xy=(0.89, 0.93), xycoords='figure fraction',
                         horizontalalignment='right', verticalalignment='top',
                         fontsize=12)

            plt.xlabel('Delta SOC [%]')
            plt.ylabel('Tiempo Viaje [minutos]')
            plt.title(f'Tiempo Viaje vs Delta SOC {ss}')

            fig.savefig(f'graficos_2d/Tiempo Viaje vs Delta SOC {ss}.png', dpi=100)
            plt.close()

    elif dibujar_pnet_dsoc:
        for ss in df['codigo_Ruta'].unique():
            print(f'Graficando 2d pnet_dsoc {ss}')
            dfx_ = df.loc[df['codigo_Ruta'] == ss].copy()
            corrS = dfx_[['Potencia_neta', 'delta_soc']].corr(method='spearman')
            corrP = dfx_[['Potencia_neta', 'delta_soc']].corr(method='pearson')

            fig = plt.figure()
            grafico = sns.scatterplot(x="delta_soc",
                                      y="Potencia_neta",
                                      hue='valor_soc_Ttec_ini',
                                      palette=cmap,
                                      data=dfx_
                                      )

            fig = grafico.get_figure()
            plt.legend(title='Soc inicial')
            plt.xlim(min_dsoc, max_dsoc)
            plt.ylim(min_pnet, max_pnet)
            plt.annotate(f'Correlacion (S): {round(corrS.iloc[0][1], 3):.3f}\n'
                         f'Correlacion (P): {round(corrP.iloc[0][1], 3):.3f}',
                         xy=(0.89, 0.93), xycoords='figure fraction',
                         horizontalalignment='right', verticalalignment='top',
                         fontsize=12)

            plt.xlabel('Delta SOC [%]')
            plt.ylabel('Potencia Consumida neta por hora [kWh]')
            plt.title(f'Potencia Consumida vs Delta SOC {ss}')

            fig.savefig(f'graficos_2d/PCN vs Delta SOC {ss}.png', dpi=100)
            plt.close()

    elif dibujar_dsoc_si:
        for ss in df['codigo_Ruta'].unique():
            print(f'Graficando 2d dsoc_si {ss}')
            dfx_ = df.loc[df['codigo_Ruta'] == ss].copy()
            corrS = dfx_[['delta_soc', 'valor_soc_Ttec_ini']].corr(method='spearman')
            corrP = dfx_[['delta_soc', 'valor_soc_Ttec_ini']].corr(method='pearson')

            fig = plt.figure()
            grafico = sns.scatterplot(x="delta_soc",
                                      y="valor_soc_Ttec_ini",
                                      hue='tiempo_viaje',
                                      palette=cmap,
                                      data=dfx_
                                      )

            fig = grafico.get_figure()
            plt.legend(title='Tviaje')
            plt.xlim(min_dsoc, max_dsoc)
            plt.ylim(min_si, max_si)
            plt.annotate(f'Correlacion (S): {round(corrS.iloc[0][1], 3):.3f}\n'
                         f'Correlacion (P): {round(corrP.iloc[0][1], 3):.3f}',
                         xy=(0.89, 0.93), xycoords='figure fraction',
                         horizontalalignment='right', verticalalignment='top',
                         fontsize=12)

            plt.xlabel('Delta SOC [%]')
            plt.ylabel('SOC Inicio expedición [%]')
            plt.title(f'SOC Inicial vs Delta SOC {ss}')

            fig.savefig(f'graficos_2d/SOC_ini vs Delta SOC {ss}.png', dpi=100)
            plt.close()


if dibujar_3d:
    dibujar_dsoc_3d = True
    dibujar_pnet_3d = False
    dibujar_pcon_3d = False
    dibujar_pgen_3d = False

    if dibujar_dsoc_3d:
        variable = 'delta_soc'
        texto_ejez = 'Delta SOC [%]'
    elif dibujar_pnet_3d:
        variable = 'Potencia_neta'
        texto_ejez = 'Potencia Consumida neta por hora [kWh]'
    elif dibujar_pcon_3d:
        variable = 'delta_Pcon'
        texto_ejez = 'Potencia Consumida [kWh]'
    elif dibujar_pgen_3d:
        variable = 'delta_Pgen'
        texto_ejez = 'Potencia Generada [kWh]'

    rutas_outlier = []
    for ruta in rutas:
        print(f'Graficando 3d {variable} {ruta}')
        for i in range(len(cortes) - 1):
            dfx = df.loc[(df['MH_inicio2'] >= cortes[i]) & (df['MH_inicio2'] < cortes[i + 1]) & (df['codigo_Ruta'] == ruta)].copy()

            if dfx.empty or len(dfx.index) < 5:
                # print(f"Poco o nada de datos {ruta} MH {cortes[i]}-{cortes[i + 1]}")
                continue

            if dibujar_pnet_3d or dibujar_pcon_3d or dibujar_pgen_3d:
                dfx['texto'] = dfx.apply(lambda x: texto_pnet(x['PPU'], x['Potencia_neta'],
                                                              x['distancia_recorrida'],
                                                              x['tiempo_viaje'],
                                                              x['MH_inicio'],
                                                              x['valor_soc_Ttec_ini']), axis=1)

                fig6 = go.Figure(layout=go.Layout(
                                 title=go.layout.Title(text=(f"Potencia {ruta}"
                                                             f" entre las {int(cortes[i] / 2)} y "
                                                             f"{int(cortes[i + 1] / 2)} horas del dia")),
                                 margin=dict(b=0, l=0, r=0, t=25)))

            elif dibujar_dsoc_3d:
                dfx['texto'] = dfx.apply(lambda x: texto_dsoc(x['PPU'], x['delta_soc'],
                                                              x['distancia_recorrida'],
                                                              x['tiempo_viaje'],
                                                              x['MH_inicio'],
                                                              x['valor_soc_Ttec_ini']), axis=1)

                fig6 = go.Figure(layout=go.Layout(
                                 title=go.layout.Title(text=(f"Delta SOC {ruta}"
                                                             f" entre las {int(cortes[i] / 2)} y "
                                                             f"{int(cortes[i + 1] / 2)} horas del dia")),
                                 margin=dict(b=0, l=0, r=0, t=25)))

            fig6.update_layout(title={'y': 0.9,
                                      'x': 0.5,
                                      'xanchor': 'center',
                                      'yanchor': 'top'})

            dfx['Outlier'] = 'circle'
            j = 0

            mhsx_ = dfx['MH_inicio2'].unique()
            mhsx_.sort()
            vale_la_pena = False

            fig6.add_trace(go.Scatter3d(x=[0],
                                        y=[0],
                                        z=[0],
                                        text='0',
                                        hoverinfo='text',
                                        mode='markers',
                                        name='0',
                                        marker=dict(size=1,
                                                    color='#ffffff'
                                                    )))

            for mh_ in mhsx_:
                dfx_ = dfx.loc[dfx['MH_inicio2'] == mh_]

                IQR = dfx_[variable].quantile(.75) - dfx_[variable].quantile(.25)
                cota = dfx_[variable].quantile(.75) + 3 * IQR
                cota2 = dfx_[variable].quantile(.75) + 5 * IQR

                dfx.loc[(dfx['MH_inicio2'] == mh_) & (dfx[variable] > cota), 'Outlier'] = 'diamond'
                dfx.loc[(dfx['MH_inicio2'] == mh_) & (dfx[variable] > cota2), 'Outlier'] = 'x'

                dfx_ = dfx.loc[dfx['MH_inicio2'] == mh_]

                # Condicion para dibujar: tener outliers "debil" (diamond) o "notorio" (x)
                if len(dfx_.loc[dfx_['Outlier'] == 'x'].index) > 2:
                    print(f'{ruta} {int(cortes[i] / 2)}_{int(cortes[i + 1] / 2)}hrs tiene outliers notorios, se va a hacer gráfico 3d')
                    vale_la_pena = True

                fig6.add_trace(go.Scatter3d(x=dfx_['distancia_recorrida'],
                                            y=dfx_['tiempo_viaje'],
                                            z=dfx_[variable],
                                            text=dfx_['texto'],
                                            hoverinfo='text',
                                            mode='markers',
                                            name=Diccionario_MH[mh_],
                                            marker=dict(size=8,
                                                        color=colores[j % len(colores)],
                                                        opacity=1,
                                                        symbol=dfx_['Outlier']
                                                        )))
                j += 1

            if vale_la_pena:

                fig6.update_layout(scene_aspectmode='manual',
                                   scene_aspectratio=dict(x=1.2, y=1.6, z=0.8),
                                   scene=dict(xaxis_title='Distancia recorrida [km]',
                                              yaxis_title='Tiempo de viaje [min]',
                                              zaxis_title=texto_ejez)
                                   )

                fig6.write_html(f'{variable}_{ruta}_{int(cortes[i] / 2)}_{int(cortes[i + 1] / 2)}hrs.html',
                                config={'scrollZoom': True, 'displayModeBar': True})

            rutas_outlier.append(dfx.loc[dfx['Outlier'].isin(['x', 'diamond']), ['Indice_mensual', 'Outlier']].copy())

    df_outlier = pd.concat(rutas_outlier)
    df_outlier.loc[df_outlier['Outlier'] == 'x', 'Outlier'] = 'Notorio'
    df_outlier.loc[df_outlier['Outlier'] == 'diamond', 'Outlier'] = 'Debil'

    df = pd.read_parquet(archivo_data)
    df = df.merge(df_outlier, on='Indice_mensual', suffixes=['', '_o'])
    df['Mes'] = df['Fecha'].dt.month
    df.to_excel(f'Outliers_{variable}.xlsx', index=False)
