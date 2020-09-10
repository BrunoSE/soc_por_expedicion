import pandas as pd
import os
import plotly.graph_objects as go
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


def graficar(variable_graficar: str):
    global df_final
    vary = [f'{variable_graficar}_25%',
            f'{variable_graficar}_50%',
            f'{variable_graficar}_75%']

    columnas_de_interes = [x for x in columnas_groupby]
    columnas_de_interes.append(variable_graficar)

    df_var = df_final.loc[~(df_final[variable_graficar].isna()), columnas_de_interes]
    df_var = df_var.groupby(by=columnas_groupby).describe().reset_index()
    df_var.columns = ['_'.join(col).rstrip('_') for col in df_var.columns.values]
    # describe entrega col_count, col_mean, col_std, col_min, col_max, col_50%, 25% y 75%
    # pasar MH a datetime en una nueva columna
    df_var = df_var.loc[df_var[f'{variable_graficar}_count'] > 2]
    df_var['Media Hora'] = df_var['MH_inicio'].map(dict_mh_date_str)

    # plotear
    marcadores = ['circle', 'square', 'diamond', 'pentagon', 'triangle-up',
                  'triangle-down', 'cross', 'hexagon']

    colores_2 = [('#ff7f00', 'rgba(9, 112, 210, 0.2)'),
                 ('#0080FF', 'rgba(0, 128, 0, 0.2)'),
                 ('#008000', 'rgba(0, 128, 0, 0.2)'),
                 ('#000000', 'rgba(0, 0, 0, 0.2)')]

    contador = 0
    df_cero = pd.DataFrame(mh_del_dia, columns=['Media Hora'])
    df_cero['Cero'] = 0
    df_cero = df_cero.loc[df_cero['Media Hora'] >= df_var['Media Hora'].min()]
    df_cero = df_cero.loc[df_cero['Media Hora'] <= df_var['Media Hora'].max()]

    # iterar servicios
    for ss in df_var['Servicio_Sentido'].unique():
        logger.info(f'Graficando {variable_graficar} {ss}')
        fig = go.Figure()
        # agregar un 0 para forzar mostrar el origen 0, 0
        fig.add_trace(go.Scatter(x=df_cero['Media Hora'],
                                 y=df_cero['Cero'],
                                 name="0",
                                 marker_color="white"))

        dfx = pd.merge(df_cero[['Media Hora']],
                       df_var.loc[df_var['Servicio_Sentido'] == ss],
                       how='left',
                       on='Media Hora')

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'], y=dfx[vary[2]],
                       name=f'percentil75',
                       mode='lines',
                       connectgaps=False,
                       opacity=0.5,
                       line_color=colores_2[contador % len(colores_2)][0]))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'], y=dfx[vary[1]],
                       name=f'Mediana',
                       mode='lines+markers',
                       connectgaps=False,
                       marker=dict(size=7, symbol=marcadores[contador % len(marcadores)]),
                       line_color=colores_2[contador % len(colores_2)][0]))

        fig.add_trace(
            go.Scatter(x=dfx['Media Hora'], y=dfx[vary[0]],
                       name=f'percentil25',
                       mode='lines',
                       connectgaps=False,
                       opacity=0.5,
                       line_color=colores_2[contador % len(colores_2)][0]))

        contador += 1

        # Set x-axis title
        fig.update_xaxes(title_text="Media Hora Despacho",
                         showticklabels=True
                         )

        # Set y-axes titles
        if variable_graficar == 'delta_soc':
            # Add figure title
            fig.update_layout(title=go.layout.Title(
                text=f"Variación en %SOC por expedición {ss}",
                font=dict(size=20, color='#000000')),
                font=dict(size=16, color='#000000'),
                xaxis_tickformat='%H:%M',
            )
            fig.update_yaxes(title_text="", tickformat=" %")

        elif variable_graficar == 'delta_Pcon':
            # Add figure title
            fig.update_layout(title=go.layout.Title(
                text=f"Potencia consumida por expedición {ss}",
                font=dict(size=20, color='#000000')),
                font=dict(size=16, color='#000000'),
                xaxis_tickformat='%H:%M',
            )
            fig.update_yaxes(title_text="Potencia [kW]")

        elif variable_graficar == 'delta_Pgen':
            # Add figure title
            fig.update_layout(title=go.layout.Title(
                text=f"Potencia generada por expedición {ss}",
                font=dict(size=20, color='#000000')),
                font=dict(size=16, color='#000000'),
                xaxis_tickformat='%H:%M',
            )
            fig.update_yaxes(title_text="Potencia [kW]")

        fig.write_html(f'grafico_{ss}_{variable_graficar}.html',
                       config={'scrollZoom': True, 'displayModeBar': True})
        fig.write_image(f'grafico_{ss}_{variable_graficar}.png', width=1600, height=800)


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
    graficar('delta_Pcon')
    graficar('delta_Pgen')
    logger.info('Listo todo')
