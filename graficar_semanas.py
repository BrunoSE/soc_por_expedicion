import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import logging
global logger
global file_format
global df_final
global primera_semana
global ultima_semana

df_final = []
# declarar diccionario con string - media hora en formato datetime
medianoche = datetime.datetime(1980, 1, 1, 0, 0, 0)
mh_del_dia = [datetime.datetime(1980, 1, 1, 0, 0, 0)]

for _ in range(1, 48):
    medianoche = medianoche + datetime.timedelta(minutes=30)
    mh_del_dia.append(medianoche)

mh_del_dia_str = [mh.strftime('%H:%M:%S') for mh in mh_del_dia]
dict_mh_date_str = dict(zip(mh_del_dia_str, mh_del_dia))

# para crear groupby
columnas_groupby = ['Servicio', 'Sentido', 'Servicio_Sentido', 'MH_inicio']

# para plotear
marcadores = ['circle', 'square', 'diamond', 'pentagon', 'triangle-up',
              'triangle-down', 'cross', 'hexagon']

colores_2 = [('#ff7f00', 'rgba(9, 112, 210, 0.9)'),
             ('#0080FF', 'rgba(0, 128, 0, 0.9)'),
             ('#008000', 'rgba(0, 128, 0, 0.9)'),
             ('#000000', 'rgba(0, 0, 0, 0.9)')]

marker_size = 10
opacity = 0.6

def mantener_log():
    global logger
    global file_format
    logger = logging.getLogger(__name__)  # P: número de proceso, L: número de línea
    logger.setLevel(logging.DEBUG)  # deja pasar todos desde debug hasta critical
    print_handler = logging.StreamHandler()
    print_format = logging.Formatter('[{asctime:s}] {levelname:s} L{lineno:d}| {message:s}',
                                     '%Y-%m-%d %H:%M:%S', style='{')
    file_format = logging.Formatter('[{asctime:s}] {processName:s} P{process:d}@{name:s} ' +
                                    '${levelname:s} L{lineno:d}| {message:s}',
                                    '%Y-%m-%d %H:%M:%S', style='{')
    # printear desde debug hasta critical:
    print_handler.setLevel(logging.DEBUG)
    print_handler.setFormatter(print_format)
    logger.addHandler(print_handler)
    return logger


def pipeline(dia_ini, mes, anno, replace_data_ttec=False, replace_resumen=False):
    global df_final
    global primera_semana
    global ultima_semana

    # Sacar fechas de interes a partir de lunes inicio de semana
    fecha_dia_ini = pd.to_datetime(f'{dia_ini}-{mes}-{anno}').date()
    dia_de_la_semana = fecha_dia_ini.isoweekday()
    if dia_de_la_semana != 1:
        logger.warning(f"Primer día no es lunes, numero: {dia_de_la_semana}")
    if dia_de_la_semana > 5:
        logger.error(f"Primer día es fin de semana, numero: {dia_de_la_semana}")
        exit()

    fechas_de_interes = []
    # se buscan días de la semana entre fecha inicio y el viernes siguiente
    for i in range(0, 6 - dia_de_la_semana):
        fechas_de_interes.append(fecha_dia_ini + pd.Timedelta(days=i))

    fechas_de_interes = [x.strftime('%Y-%m-%d') for x in fechas_de_interes]

    # Crear variable que escribe en log file de este dia
    el_dia_fin = fechas_de_interes[-1].split('-')[-1]
    nombre_semana = f"semana_{fechas_de_interes[0].replace('-', '_')}_{el_dia_fin}"

    # buscar si ya existia carpeta
    if not os.path.isdir(nombre_semana):
        logger.error(f"No existe carpeta {nombre_semana}")
        exit()

    # lectura de data y creacion de tablas dinamicas con groupby
    logger.info(f'Leyendo ./{nombre_semana}/dataf_{nombre_semana}.parquet')
    df = pd.read_parquet(f'./{nombre_semana}/dataf_{nombre_semana}.parquet')

    # filtrar
    df = df.loc[(df['Operativo'] == 'C')]
    df = df.loc[(df['Cumple_Triada_Revisada'] == 1)]
    df['pctje_dist_recorrida'] = df['distancia_recorrida'] / df['dist_Ruta']
    df = df.loc[df.pctje_dist_recorrida > 0.85]
    df = df.loc[df.pctje_dist_recorrida < 1.15]

    if not df_final:
        primera_semana = nombre_semana

    df_final.append(df)
    ultima_semana = nombre_semana


def graficar(variable_graficar: str, filtrar_outliers_intercuartil: bool = True):
    global df_final
    vary = [f'{variable_graficar}_25%',
            f'{variable_graficar}_50%',
            f'{variable_graficar}_75%',
            f'{variable_graficar}_count']

    columnas_de_interes = [x for x in columnas_groupby]
    columnas_de_interes.append(variable_graficar)

    df_fv = df_final.loc[~(df_final[variable_graficar].isna()), columnas_de_interes]
    # describe entrega col_count, col_mean, col_std, col_min, col_max, col_50%, 25% y 75%
    df_var = df_fv.groupby(by=columnas_groupby).describe().reset_index()
    df_var.columns = ['_'.join(col).rstrip('_') for col in df_var.columns.values]
    # filtrar MH con menos de 3 datos
    df_var = df_var.loc[df_var[f'{variable_graficar}_count'] > 2]

    if filtrar_outliers_intercuartil:
        df_var['IQR'] = df_var[vary[2]] - df_var[vary[0]]
        df_var['cota_inf'] = df_var[vary[0]] - 1.5 * df_var['IQR']
        df_var['cota_sup'] = df_var[vary[2]] + 1.5 * df_var['IQR']
        for row in zip(df_var['MH_inicio'], df_var['Servicio_Sentido'],
                       df_var['cota_inf'], df_var['cota_sup']):
            select1 = ((df_fv['MH_inicio'] == row[0]) & (df_fv['Servicio_Sentido'] == row[1]))
            select2 = ((df_fv[variable_graficar] >= row[2]) & (df_fv[variable_graficar] <= row[3]))
            df_fv = df_fv.loc[((select1 & select2) | (~select1))]

        df_var = df_fv.groupby(by=columnas_groupby).describe().reset_index()
        df_var.columns = ['_'.join(col).rstrip('_') for col in df_var.columns.values]
        # filtrar MH con menos de 2 datos
        df_var = df_var.loc[df_var[f'{variable_graficar}_count'] > 2]

    # pasar MH a datetime en una nueva columna
    df_var['Media Hora'] = df_var['MH_inicio'].map(dict_mh_date_str)

    # plotear
    contador = 0
    df_cero = pd.DataFrame(mh_del_dia, columns=['Media Hora'])
    df_cero['Cero'] = 0
    df_cero = df_cero.loc[df_cero['Media Hora'] >= df_var['Media Hora'].min()]
    df_cero = df_cero.loc[df_cero['Media Hora'] <= df_var['Media Hora'].max()]
    nombre_cero = '0'
    if filtrar_outliers_intercuartil:
        nombre_cero = 's0'

    # iterar servicios
    for ss in df_var['Servicio_Sentido'].unique():
        el_color = colores_2[contador % len(colores_2)][0]
        logger.info(f'Graficando {variable_graficar} {ss}')
        fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15])
        # agregar un 0 para forzar mostrar el origen 0, 0
        fig.add_trace(go.Scatter(x=df_cero['Media Hora'].dt.time,
                                 y=df_cero['Cero'],
                                 name=nombre_cero,
                                 marker_color="white"))

        dfx = pd.merge(df_cero[['Media Hora']],
                       df_var.loc[df_var['Servicio_Sentido'] == ss],
                       how='left',
                       on='Media Hora')

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary[2]],
                       name=f'percentil75',
                       mode='lines',
                       connectgaps=True,
                       opacity=opacity,
                       line_color=el_color))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary[1]],
                       name=f'Mediana',
                       mode='lines+markers',
                       connectgaps=True,
                       marker=dict(size=marker_size,
                                   symbol=marcadores[contador % len(marcadores)]),
                       line_color=el_color))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary[0]],
                       name=f'percentil25',
                       mode='lines',
                       connectgaps=True,
                       opacity=opacity,
                       line_color=el_color))

        # Agregar bar plot abajo con numero de datos
        # width=[0.5] * len(dfx['Media Hora'].index),
        fig.add_trace(go.Bar(x=dfx['Media Hora'].dt.time, y=dfx[vary[3]],
                             opacity=0.95, name='Nro Datos'),
                      row=2, col=1)

        # Set y-axes titles
        if variable_graficar == 'delta_soc':
            # Add figure title
            fig.update_layout(title=go.layout.Title(
                text=f"Variación en %SOC por expedición {ss}",
                font=dict(size=20, color='#000000')),
                font=dict(size=16, color='#000000'),
                xaxis_tickformat='%H:%M',
                bargap=0.3
            )
            fig.update_yaxes(title_text="", tickformat=" %", row=1, col=1)

        elif variable_graficar == 'delta_Pcon':
            # Add figure title
            fig.update_layout(title=go.layout.Title(
                text=f"Potencia consumida por expedición {ss}",
                font=dict(size=20, color='#000000')),
                font=dict(size=16, color='#000000'),
                xaxis_tickformat='%H:%M'
            )
            fig.update_yaxes(title_text="Potencia [kW]", row=1, col=1)

        elif variable_graficar == 'delta_Pgen':
            # Add figure title
            fig.update_layout(title=go.layout.Title(
                text=f"Potencia generada por expedición {ss}",
                font=dict(size=20, color='#000000')),
                font=dict(size=14, color='#000000'),
                xaxis_tickformat='%H:%M'
            )
            fig.update_yaxes(title_text="Potencia [kW]", row=1, col=1)

        # Set x-axis title
        fig.update_xaxes(title_text="Numero de datos por media hora despacho",
                         showticklabels=False,
                         type='category',
                         row=2, col=1
                         )
        fig.update_xaxes(
                         showticklabels=True,
                         type='category',
                         row=1, col=1
                         )

        if filtrar_outliers_intercuartil:
            fig.write_html(f'graf_{ss}_{variable_graficar}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'graf_{ss}_{variable_graficar}.png', width=1600, height=800)
        else:
            fig.write_html(f'grafico_{ss}_{variable_graficar}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'grafico_{ss}_{variable_graficar}.png', width=1600, height=800)

        contador += 1
        exit()


def graficar_potencias():
    global df_final
    variable_graficar = 'delta_Pcon'
    variable_graficar2 = 'delta_Pgen'
    vary = [f'{variable_graficar}_25%',
            f'{variable_graficar}_50%',
            f'{variable_graficar}_75%']

    vary2 = [f'{variable_graficar2}_25%',
             f'{variable_graficar2}_50%',
             f'{variable_graficar2}_75%']

    columnas_de_interes = [x for x in columnas_groupby]
    columnas_de_interes.append(variable_graficar)

    columnas_de_interes2 = [x for x in columnas_groupby]
    columnas_de_interes2.append(variable_graficar2)

    df_var = df_final.loc[~(df_final[variable_graficar].isna()), columnas_de_interes]
    df_var = df_var.groupby(by=columnas_groupby).describe().reset_index()
    df_var.columns = ['_'.join(col).rstrip('_') for col in df_var.columns.values]
    # describe entrega col_count, col_mean, col_std, col_min, col_max, col_50%, 25% y 75%
    # pasar MH a datetime en una nueva columna
    df_var = df_var.loc[df_var[f'{variable_graficar}_count'] > 2]
    df_var['Media Hora'] = df_var['MH_inicio'].map(dict_mh_date_str)

    df_var2 = df_final.loc[~(df_final[variable_graficar2].isna()), columnas_de_interes2]
    df_var2 = df_var2.groupby(by=columnas_groupby).describe().reset_index()
    df_var2.columns = ['_'.join(col).rstrip('_') for col in df_var2.columns.values]
    # describe entrega col_count, col_mean, col_std, col_min, col_max, col_50%, 25% y 75%
    # pasar MH a datetime en una nueva columna
    df_var2 = df_var2.loc[df_var2[f'{variable_graficar2}_count'] > 2]
    df_var2['Media Hora'] = df_var2['MH_inicio'].map(dict_mh_date_str)

    # plotear
    contador = 0
    df_cero = pd.DataFrame(mh_del_dia, columns=['Media Hora'])
    df_cero['Cero'] = 0
    df_cero = df_cero.loc[df_cero['Media Hora'] >= df_var['Media Hora'].min()]
    df_cero = df_cero.loc[df_cero['Media Hora'] <= df_var['Media Hora'].max()]

    # iterar servicios
    for ss in df_var['Servicio_Sentido'].unique():
        logger.info(f'Graficando {variable_graficar} y {variable_graficar2} {ss}')
        fig = go.Figure()
        # agregar un 0 para forzar mostrar el origen 0, 0
        fig.add_trace(go.Scatter(x=df_cero['Media Hora'].dt.time,
                                 y=df_cero['Cero'],
                                 name="0",
                                 marker_color="white"))

        dfx = pd.merge(df_cero[['Media Hora']],
                       df_var.loc[df_var['Servicio_Sentido'] == ss],
                       how='left',
                       on='Media Hora')

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary[2]],
                       name=f'PC_percentil75',
                       mode='lines',
                       connectgaps=True,
                       opacity=opacity,
                       line_color=colores_2[contador % len(colores_2)][0]))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary[1]],
                       name=f'PCons_Mediana',
                       mode='lines+markers',
                       connectgaps=True,
                       marker=dict(size=marker_size,
                                   symbol=marcadores[contador % len(marcadores)]),
                       line_color=colores_2[contador % len(colores_2)][0]))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary[0]],
                       name=f'PC_percentil25',
                       mode='lines',
                       connectgaps=True,
                       opacity=opacity,
                       line_color=colores_2[contador % len(colores_2)][0]))

        contador += 1
        dfx = pd.merge(df_cero[['Media Hora']],
                       df_var2.loc[df_var2['Servicio_Sentido'] == ss],
                       how='left',
                       on='Media Hora')

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary2[2]],
                       name=f'PG_percentil75',
                       mode='lines',
                       connectgaps=True,
                       opacity=opacity,
                       line_color=colores_2[contador % len(colores_2)][0]))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary2[1]],
                       name=f'PG_Mediana',
                       mode='lines+markers',
                       connectgaps=True,
                       marker=dict(size=marker_size,
                                   symbol=marcadores[contador % len(marcadores)]),
                       line_color=colores_2[contador % len(colores_2)][0]))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[vary2[0]],
                       name=f'PGen_percentil25',
                       mode='lines',
                       connectgaps=True,
                       opacity=opacity,
                       line_color=colores_2[contador % len(colores_2)][0]))

        contador += 1

        # Add figure title
        fig.update_layout(xaxis_tickformat='%H:%M',
                          font=dict(size=16, color='#000000'),
                          title=go.layout.Title(
                              text=f"Potencia Consumida y Generada por expedición {ss}",
                              font=dict(size=20, color='#000000'))
                          )
        # Set y-axes titles
        fig.update_yaxes(title_text="Potencia [kW]")

        # Set x-axis title
        fig.update_xaxes(title_text="Media Hora Despacho",
                         showticklabels=True,
                         type='category'
                         )

        fig.write_html(f'grafico_{ss}_Potencias.html',
                       config={'scrollZoom': True, 'displayModeBar': True}
                       )
        fig.write_image(f'grafico_{ss}_Potencias.png',
                        width=1600, height=800)


if __name__ == '__main__':
    global primera_semana
    global ultima_semana

    logger = mantener_log()

    reemplazar_data_ttec = False
    reemplazar_resumen = False
    pipeline(17, 8, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(24, 8, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(31, 8, 2020, reemplazar_data_ttec, reemplazar_resumen)

    logger.info('Graficando')
    df_final = pd.concat(df_final)
    df_final['delta_soc'] = df_final['delta_soc'] * 0.01
    df_final['delta_Pcon'] = df_final['delta_Pcon'] * 0.001
    df_final['delta_Pgen'] = df_final['delta_Pgen'] * 0.001

    sem_primera = primera_semana.replace('semana_', '')[:-3]
    sem_ultima = ultima_semana.replace('semana_', '')[:-3]
    carpeta_guardar_graficos = f'graficos_{sem_primera}_{sem_ultima}'

    if not os.path.isdir(carpeta_guardar_graficos):
        logger.info(f'Creando carpeta {carpeta_guardar_graficos}')
        os.mkdir(carpeta_guardar_graficos)
    else:
        logger.warning(f'Reescribiendo sobre carpeta {carpeta_guardar_graficos}')

    os.chdir(carpeta_guardar_graficos)
    graficar('delta_soc')
    # graficar_potencias()
    logger.info('Listo todo')
