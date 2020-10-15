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
minimos_datos_por_mh = 1

# para plotear
marcadores = ['circle', 'square', 'diamond', 'pentagon', 'triangle-up',
              'triangle-down', 'cross', 'hexagon']

colores_2 = [('#ff7f00', 'rgba(9, 112, 210, 0.5)'),
             ('#0080FF', 'rgba(0, 128, 0, 0.5)'),
             ('#008000', 'rgba(0, 128, 0, 0.5)'),
             ('#000000', 'rgba(0, 0, 0, 0.5)')]

colorLineas_ejeYppal = 'rgb(200, 200, 200)'
marker_size = 11
opacity_percentiles = 0.7
opacity_barras = 0.6
ancho_barras = 0.5
ticks_para_barras = [minimos_datos_por_mh, 5, 15]
texto_ticks_barras = [str(x) for x in ticks_para_barras]
zoom_out_barras = 1.5  # mas grande implica barras de conteo mas pequeñas


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


def g_pipeline(dia_ini, mes, anno, sem_especial=[], tipo_dia):
    global df_final
    global primera_semana
    global ultima_semana
    # dia_ini tiene que ser un día lunes
    # Sacar fechas de interes a partir de lunes inicio de semana
    fecha_dia_ini = pd.to_datetime(f'{dia_ini}-{mes}-{anno}', dayfirst=True).date()
    dia_de_la_semana = fecha_dia_ini.isoweekday()
    if dia_de_la_semana != 1:
        logger.error(f"Primer día no es lunes y se quiere ocupar parámetro sem_especial, "
                     f"numero dia_ini: {dia_de_la_semana}")
        exit()

    fechas_de_interes_dt = []
    if not sem_especial:
        for i in range(0, 7):
            fechas_de_interes_dt.append(fecha_dia_ini + pd.Timedelta(days=i))
    else:
        # se buscan días de la semana entre fecha inicio y el domingo
        if len(sem_especial) != len(set(sem_especial)):
            logger.error(f"Semana especial no debe repetir números: {sem_especial}")
            exit()
        for i in sem_especial:
            if 0 < i < 8:
                fechas_de_interes_dt.append(fecha_dia_ini + pd.Timedelta(days=(i - 1)))
            else:
                logger.error(f"Semana especial debe ser lista con números 1 al 7: {sem_especial}")
                exit()
    fechas_de_interes = [x.strftime('%Y-%m-%d') for x in fechas_de_interes_dt]

    logger.info(f'Semana de interes: {fechas_de_interes}')

    # Crear variable que escribe en log file de este dia
    nombre_semana = f"semana_{fechas_de_interes[0].replace('-', '_')}"

    # buscar si ya existia carpeta
    if not os.path.isdir(nombre_semana):
        logger.error(f"No existe carpeta {nombre_semana}")
        exit()

    # sacar las fechas con el tipo de dia buscado
    fecha_util = []
    if tipo_dia == 'Laboral':
        fecha_util = [fecha_ for fecha_ in fechas_de_interes_dt if fecha_.isoweekday() < 6]
    elif tipo_dia == 'Sabado' or tipo_dia == 'Sábado':
        fecha_util = [fecha_ for fecha_ in fechas_de_interes_dt if fecha_.isoweekday() == 6]
    elif tipo_dia == 'Domingo':
        fecha_util = [fecha_ for fecha_ in fechas_de_interes_dt if fecha_.isoweekday() == 7]
    else:
        logger.error("Variable tipo_dia tiene que ser 'Laboral' o 'Sabado' o 'Domingo'")
        exit()

    if fecha_util:
        fechas_de_interes = [x.strftime('%Y_%m_%d') for x in fecha_util]
        logger.info(f'Dias {tipo_dia} de la semana: {fechas_de_interes}')
    else:
        logger.warning(f'Semana {nombre_semana} no tiene dias tipo {tipo_dia}')
        return

    # lectura de data y creacion de tablas dinamicas con groupby
    df = []
    for fecha_ in fechas_de_interes:
        logger.info(f'Leyendo ./{nombre_semana}/data_196rE_{fecha_}.parquet')
        df.append(pd.read_parquet(f'./{nombre_semana}/data_196rE_{fecha_}.parquet'))

    df = pd.concat(df)
    df = df.loc[df['delta_soc'] > 0]
    if df.empty:
        logger.warning('Este día no tiene data tracktec')
    else:
        # filtrar
        df = df.loc[(df['Operativo'] == 'C')]
        df = df.loc[(df['Cumple_Triada_Revisada'] == 1)]
        df['pctje_dist_recorrida'] = df['distancia_recorrida'] / df['dist_Ruta']
        df = df.loc[df['pctje_dist_recorrida'] > 0.85]
        df = df.loc[df['pctje_dist_recorrida'] < 1.15]
        df = df.loc[df['d_registros_ini'] < 1000]
        df = df.loc[df['d_registros_fin'] < 1000]

        # Transformar soc a porcentaje y potencias a kW
        df['delta_soc'] = df['delta_soc'] * 0.01
        df['delta_Pcon'] = df['delta_Pcon'] * 0.001
        df['delta_Pgen'] = df['delta_Pgen'] * 0.001

        df = df.loc[df['delta_Pcon'] > 0]
        df = df.loc[df['delta_Pgen'] > 0]

    if not df_final:
        primera_semana = nombre_semana

    df_final.append(df)
    ultima_semana = nombre_semana


def graficar_boxplot(variable_graficar: str, filtrar_outliers_intercuartil: bool = True,
                     tipo_dia='Laboral', nombre=''):
    # para cada ss grafica mediana y percentiles 25 y 75 por mh de una variable
    if not os.path.isdir(f'{variable_graficar}_{tipo_dia}'):
        logger.info(f'Creando carpeta {variable_graficar}_{tipo_dia}')
        os.mkdir(f'{variable_graficar}_{tipo_dia}')
    else:
        logger.warning(f'Reescribiendo sobre carpeta {variable_graficar}_{tipo_dia}')

    os.chdir(f'{variable_graficar}_{tipo_dia}')

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
    df_var = df_var.loc[df_var[f'{variable_graficar}_count'] >= minimos_datos_por_mh]

    if filtrar_outliers_intercuartil:
        df_var['IQR'] = df_var[vary[2]] - df_var[vary[0]]
        df_var['cota_inf'] = df_var[vary[0]] - 1.5 * df_var['IQR']
        df_var['cota_sup'] = df_var[vary[2]] + 1.5 * df_var['IQR']
        for row in zip(df_var['MH_inicio'], df_var['Servicio_Sentido'],
                       df_var['cota_inf'], df_var['cota_sup']):
            select1 = ((df_fv['MH_inicio'] == row[0]) & (df_fv['Servicio_Sentido'] == row[1]))
            select2 = ((df_fv[variable_graficar] >= row[2]) & (df_fv[variable_graficar] <= row[3]))
            df_fv = df_fv.loc[((select1 & select2) | (~select1))]

    df_fv2 = df_fv.copy()
    for ss in df_fv2['Servicio_Sentido'].unique():
        for mh in df_fv2.loc[df_fv2['Servicio_Sentido'] == ss, 'MH_inicio'].unique():
            if len(df_fv2.loc[((df_fv2['Servicio_Sentido'] == ss) &
                               (df_fv2['MH_inicio'] == mh))].index) < minimos_datos_por_mh:
                df_fv = df_fv.loc[((df_fv['Servicio_Sentido'] != ss) | (df_fv['MH_inicio'] != mh))]
    # pasar MH a datetime en una nueva columna
    df_fv['Media Hora'] = df_fv['MH_inicio'].map(dict_mh_date_str)
    contador = 0
    max_data_vary = df_fv[f'{variable_graficar}'].max() + 0.005
    df_fv = df_fv.sort_values(by=['Media Hora', 'Servicio_Sentido'])

    df_cero = pd.DataFrame(mh_del_dia, columns=['Media Hora'])
    df_cero['Cero'] = 0
    df_cero = df_cero.loc[df_cero['Media Hora'] >= df_fv['Media Hora'].min()]
    df_cero = df_cero.loc[df_cero['Media Hora'] <= df_fv['Media Hora'].max()]
    nombre_cero = '0'
    if filtrar_outliers_intercuartil:
        nombre_cero = 's0'

    nombre_ = ''
    if nombre:
        nombre_ = nombre.replace('-', '_')

    for ss in df_fv['Servicio_Sentido'].unique():
        el_color = colores_2[contador % len(colores_2)][0]
        logger.info(f'Graficando boxplot {variable_graficar} {ss}')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cero['Media Hora'].dt.time,
                                 y=df_cero['Cero'],
                                 name=nombre_cero,
                                 marker_color="white"))

        df_fv2 = df_fv.loc[df_fv['Servicio_Sentido'] == ss, [variable_graficar, 'Media Hora']]
        fig.add_trace(go.Box(x=df_fv2['Media Hora'].dt.time,
                             y=df_fv2[variable_graficar],
                             notched=False,
                             name='Boxplot',
                             boxpoints=False,
                             marker_color=el_color))

        contador += 1
        # Set x-axis title
        fig.update_xaxes(showticklabels=True,
                         tickangle=270
                         )

        texto_titulo = f"Variación en %SOC por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
        if variable_graficar == 'delta_soc':
            fig.update_yaxes(title_text="", tickformat=".1%",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal)

        elif variable_graficar == 'delta_Pcon':
            texto_titulo = f"Potencia consumida por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="Potencia [kW]",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal)

        elif variable_graficar == 'delta_Pgen':
            texto_titulo = f"Potencia generada por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="Potencia [kW]",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal)

        # Add figure title
        fig.update_layout(title=go.layout.Title(
            text=texto_titulo,
            font=dict(size=20, color='#000000')),
            font=dict(size=14, color='#000000'),
            xaxis_tickformat='%H:%M',
            showlegend=False
        )

        if filtrar_outliers_intercuartil:
            fig.write_html(f'Boxplot_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'Boxplot_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)
        else:
            fig.write_html(f'BoxplotCO_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'BoxplotCO_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)

    os.chdir('..')


def graficar(variable_graficar: str, filtrar_outliers_intercuartil: bool = True,
             tipo_dia='Laboral', nombre=''):
    # para cada ss grafica mediana y percentiles 25 y 75 por mh de una variable
    if not os.path.isdir(f'{variable_graficar}_{tipo_dia}'):
        logger.info(f'Creando carpeta {variable_graficar}_{tipo_dia}')
        os.mkdir(f'{variable_graficar}_{tipo_dia}')
    else:
        logger.warning(f'Reescribiendo sobre carpeta {variable_graficar}_{tipo_dia}')

    os.chdir(f'{variable_graficar}_{tipo_dia}')

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
    df_var = df_var.loc[df_var[f'{variable_graficar}_count'] >= minimos_datos_por_mh]

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
        df_var = df_var.loc[df_var[f'{variable_graficar}_count'] >= minimos_datos_por_mh]

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

    nombre_ = ''
    if nombre:
        nombre_ = nombre.replace('-', '_')

    max_data_count = max(df_var[vary[3]].max(), 50)
    max_data_vary = df_var[vary[2]].max() + 0.005

    # iterar servicios
    for ss in df_var['Servicio_Sentido'].unique():
        el_color = colores_2[contador % len(colores_2)][0]
        logger.info(f'Graficando {variable_graficar} {ss}')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
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
                       opacity=opacity_percentiles,
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
                       opacity=opacity_percentiles,
                       line_color=el_color))

        # Agregar bar plot abajo con numero de datos
        fig.add_trace(go.Bar(x=dfx['Media Hora'].dt.time, y=dfx[vary[3]],
                             marker=dict(color=el_color),
                             opacity=opacity_barras,
                             name='Nro Datos',
                             width=[ancho_barras] * len(dfx['Media Hora'].index)),
                      secondary_y=True)

        # Formatear eje y secundario
        fig.update_yaxes(title_text="",
                         range=[0, int(max_data_count * zoom_out_barras)],
                         showgrid=True,
                         showticklabels=True,
                         tickmode='array',
                         tickvals=ticks_para_barras,
                         ticktext=texto_ticks_barras,
                         secondary_y=True)

        # Set x-axis title
        fig.update_xaxes(title_text="Nro Datos - Media hora despacho",
                         showticklabels=True,
                         type='category',
                         tickangle=270
                         )

        texto_titulo = f"Variación en %SOC por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
        if variable_graficar == 'delta_soc':
            fig.update_yaxes(title_text="", tickformat=".1%",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal,
                             secondary_y=False)

        elif variable_graficar == 'delta_Pcon':
            texto_titulo = f"Potencia consumida por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="Potencia [kW]",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal,
                             secondary_y=False)

        elif variable_graficar == 'delta_Pgen':
            texto_titulo = f"Potencia generada por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="Potencia [kW]",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal,
                             secondary_y=False)

        elif variable_graficar == 'tiempo_viaje':
            texto_titulo = f"Tiempo de viaje {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="[minutos]",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal,
                             secondary_y=False)

        # Add figure title
        fig.update_layout(title=go.layout.Title(
            text=texto_titulo,
            font=dict(size=20, color='#000000')),
            font=dict(size=14, color='#000000'),
            xaxis_tickformat='%H:%M'
        )

        if filtrar_outliers_intercuartil:
            fig.write_html(f'graf_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'graf_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)
        else:
            fig.write_html(f'grafico_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'grafico_{ss}_{variable_graficar}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)

        contador += 1
    os.chdir('..')


def graficar_potencias_2(variable_graficar: str, variable_graficar_2: str,
                         filtrar_outliers_intercuartil: bool = True,
                         tipo_dia='Laboral', nombre=''):
    # para cada ss grafica mediana y percentiles 25 y 75 por mh de dos variables
    if not os.path.isdir(f'{variable_graficar}_{variable_graficar_2}_{tipo_dia}'):
        logger.info(f'Creando carpeta {variable_graficar}_{variable_graficar_2}_{tipo_dia}')
        os.mkdir(f'{variable_graficar}_{variable_graficar_2}_{tipo_dia}')
    else:
        logger.warning(f'Reescribiendo sobre carpeta {variable_graficar}_{variable_graficar_2}_{tipo_dia}')

    os.chdir(f'{variable_graficar}_{variable_graficar_2}_{tipo_dia}')

    global df_final
    dict_leyenda = {'delta_Pcon': 'PCon',
                    'delta_Pgen': 'PGen'}

    vary = [f'{variable_graficar}_25%',
            f'{variable_graficar}_50%',
            f'{variable_graficar}_75%',
            f'{variable_graficar}_count']

    vary_2 = [f'{variable_graficar_2}_25%',
              f'{variable_graficar_2}_50%',
              f'{variable_graficar_2}_75%',
              f'{variable_graficar_2}_count']

    columnas_de_interes = [x for x in columnas_groupby]
    columnas_de_interes.append(variable_graficar)
    columnas_de_interes_2 = [x for x in columnas_groupby]
    columnas_de_interes_2.append(variable_graficar_2)

    a_vgrafricar = [variable_graficar, variable_graficar_2]
    a_vary = [vary, vary_2]

    # observar que hay un supuesto implícito:
    # que es válido trabajar con columnas que tengan NA en la otra variable
    df_fv = [df_final.loc[~(df_final[variable_graficar].isna()), columnas_de_interes],
             df_final.loc[~(df_final[variable_graficar_2].isna()), columnas_de_interes_2]]
    df_var = []

    for i in [0, 1]:
        # describe entrega col_count, col_mean, col_std, col_min, col_max, col_50%, 25% y 75%
        df_vari = df_fv[i].groupby(by=columnas_groupby).describe().reset_index()
        df_vari.columns = ['_'.join(col).rstrip('_') for col in df_vari.columns.values]
        # filtrar MH con menos de 3 datos
        df_vari = df_vari.loc[df_vari[f'{a_vgrafricar[i]}_count'] >= minimos_datos_por_mh]

        if filtrar_outliers_intercuartil:
            df_vari['IQR'] = df_vari[a_vary[i][2]] - df_vari[a_vary[i][0]]
            df_vari['cota_inf'] = df_vari[a_vary[i][0]] - 1.5 * df_vari['IQR']
            df_vari['cota_sup'] = df_vari[a_vary[i][2]] + 1.5 * df_vari['IQR']
            for row in zip(df_vari['MH_inicio'], df_vari['Servicio_Sentido'],
                           df_vari['cota_inf'], df_vari['cota_sup']):
                select1 = ((df_fv[i]['MH_inicio'] == row[0]) &
                           (df_fv[i]['Servicio_Sentido'] == row[1]))
                select2 = ((df_fv[i][a_vgrafricar[i]] >= row[2]) &
                           (df_fv[i][a_vgrafricar[i]] <= row[3]))
                df_fv[i] = df_fv[i].loc[((select1 & select2) | (~select1))]

            df_vari = df_fv[i].groupby(by=columnas_groupby).describe().reset_index()
            df_vari.columns = ['_'.join(col).rstrip('_') for col in df_vari.columns.values]
            # filtrar MH con menos de 2 datos
            df_vari = df_vari.loc[df_vari[f'{a_vgrafricar[i]}_count'] >= minimos_datos_por_mh]

        # pasar MH a datetime en una nueva columna
        df_vari['Media Hora'] = df_vari['MH_inicio'].map(dict_mh_date_str)
        df_var.append(df_vari.copy())

    # plotear
    contador = 0
    df_cero = pd.DataFrame(mh_del_dia, columns=['Media Hora'])
    df_cero['Cero'] = 0
    hra_min = min(df_var[0]['Media Hora'].min(), df_var[1]['Media Hora'].min())
    hra_max = max(df_var[0]['Media Hora'].max(), df_var[1]['Media Hora'].max())
    df_cero = df_cero.loc[df_cero['Media Hora'] >= hra_min]
    df_cero = df_cero.loc[df_cero['Media Hora'] <= hra_max]
    nombre_cero = '0'
    if filtrar_outliers_intercuartil:
        nombre_cero = 's0'

    nombre_ = ''
    if nombre:
        nombre_ = nombre.replace('-', '_')

    max_data_count = max(df_var[0][a_vary[0][3]].max(), df_var[1][a_vary[1][3]].max(), 50)
    max_data_vary = max(df_var[0][a_vary[0][2]].max(), df_var[1][a_vary[1][2]].max()) + 1

    # iterar servicios
    for ss in df_var[0]['Servicio_Sentido'].unique():
        if ss not in df_var[1]['Servicio_Sentido'].unique():
            logger.warning(f'{ss} tiene datos de {variable_graficar} pero '
                           f'no de {variable_graficar_2}')

        logger.info(f'Graficando {variable_graficar} y {variable_graficar_2} {ss}')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # agregar un 0 para forzar mostrar el origen 0, 0
        fig.add_trace(go.Scatter(x=df_cero['Media Hora'].dt.time,
                                 y=df_cero['Cero'],
                                 name=nombre_cero,
                                 marker_color="white"))

        for i in [0, 1]:
            el_color = colores_2[contador % len(colores_2)][0]
            dfx = pd.merge(df_cero[['Media Hora']],
                           df_var[i].loc[df_var[i]['Servicio_Sentido'] == ss],
                           how='left',
                           on='Media Hora')

            fig.add_trace(
                go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][2]],
                           name=f'percentil75',
                           mode='lines',
                           connectgaps=True,
                           opacity=opacity_percentiles,
                           line_color=el_color))

            fig.add_trace(
                go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][1]],
                           name=f'Mediana {dict_leyenda[a_vgrafricar[i]]}',
                           mode='lines+markers',
                           connectgaps=True,
                           marker=dict(size=marker_size,
                                       symbol=marcadores[contador % len(marcadores)]),
                           line_color=el_color))

            fig.add_trace(
                go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][0]],
                           name=f'percentil25',
                           mode='lines',
                           connectgaps=True,
                           opacity=opacity_percentiles,
                           line_color=el_color))

            # Agregar bar plot abajo con numero de datos
            fig.add_trace(go.Bar(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][3]],
                                 marker=dict(color=el_color),
                                 opacity=opacity_barras,
                                 name=f'Nro Datos {dict_leyenda[a_vgrafricar[i]]}',
                                 width=[ancho_barras * 0.6] * len(dfx['Media Hora'].index)),
                          secondary_y=True)

            contador += 1

        # Formatear eje y secundario
        fig.update_yaxes(title_text="",
                         range=[0, int(max_data_count * zoom_out_barras)],
                         showgrid=True,
                         showticklabels=True,
                         tickmode='array',
                         tickvals=ticks_para_barras,
                         ticktext=texto_ticks_barras,
                         secondary_y=True)

        # Set x-axis title
        fig.update_xaxes(title_text="Nro Datos - Media hora despacho",
                         showticklabels=True,
                         type='category',
                         tickangle=270
                         )

        texto_titulo = ""
        if ((variable_graficar == 'delta_Pcon' and variable_graficar_2 == 'delta_Pgen') or
                (variable_graficar == 'delta_Pgen' and variable_graficar_2 == 'delta_Pcon')):
            texto_titulo = f"Potencia consumida y generada por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="Potencia [kW]",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal,
                             secondary_y=False)
        else:
            texto_titulo = f"{variable_graficar} y {variable_graficar_2} por expedición {ss} (Dias {tipo_dia} de semana {nombre})"
            fig.update_yaxes(title_text="",
                             range=[0, max_data_vary],
                             gridcolor=colorLineas_ejeYppal,
                             secondary_y=False)

        # Add figure title
        fig.update_layout(title=go.layout.Title(
            text=texto_titulo,
            font=dict(size=20, color='#000000')),
            font=dict(size=14, color='#000000'),
            xaxis_tickformat='%H:%M'
        )

        if filtrar_outliers_intercuartil:
            fig.write_html(f'graf_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'graf_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)
        else:
            fig.write_html(f'grafico_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'grafico_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)

    os.chdir('..')


def graficar_soc_tv(variable_graficar: str = 'delta_soc',
                    variable_graficar_2: str = 'tiempo_viaje',
                    filtrar_outliers_intercuartil: bool = True,
                    incluir_p75y25: bool = False, tipo_dia='Laboral', nombre=''):
    # para cada ss grafica mediana y percentiles 25 y 75 por mh de dos variables
    if not os.path.isdir(f'{variable_graficar}_{variable_graficar_2}_{tipo_dia}'):
        logger.info(f'Creando carpeta {variable_graficar}_{variable_graficar_2}_{tipo_dia}')
        os.mkdir(f'{variable_graficar}_{variable_graficar_2}_{tipo_dia}')
    else:
        logger.warning(f'Reescribiendo sobre carpeta {variable_graficar}_{variable_graficar_2}_{tipo_dia}')

    os.chdir(f'{variable_graficar}_{variable_graficar_2}_{tipo_dia}')

    global df_final
    dict_leyenda = {'delta_soc': 'Delta %SOC',
                    'tiempo_viaje': 'T_Viaje [min]'}

    vary = [f'{variable_graficar}_25%',
            f'{variable_graficar}_50%',
            f'{variable_graficar}_75%',
            f'{variable_graficar}_count']

    vary_2 = [f'{variable_graficar_2}_25%',
              f'{variable_graficar_2}_50%',
              f'{variable_graficar_2}_75%',
              f'{variable_graficar_2}_count']

    columnas_de_interes = [x for x in columnas_groupby]
    columnas_de_interes.append(variable_graficar)
    columnas_de_interes_2 = [x for x in columnas_groupby]
    columnas_de_interes_2.append(variable_graficar_2)

    a_vgrafricar = [variable_graficar, variable_graficar_2]
    a_vary = [vary, vary_2]

    # observar que hay un supuesto implícito:
    # que es válido trabajar con columnas que tengan NA en la otra variable
    df_fv = [df_final.loc[~(df_final[variable_graficar].isna()), columnas_de_interes],
             df_final.loc[~(df_final[variable_graficar_2].isna()), columnas_de_interes_2]]

    df_var = []

    for i in [0, 1]:
        # describe entrega col_count, col_mean, col_std, col_min, col_max, col_50%, 25% y 75%
        df_vari = df_fv[i].groupby(by=columnas_groupby).describe().reset_index()
        df_vari.columns = ['_'.join(col).rstrip('_') for col in df_vari.columns.values]
        # filtrar MH con menos de 3 datos
        df_vari = df_vari.loc[df_vari[f'{a_vgrafricar[i]}_count'] >= minimos_datos_por_mh]

        if filtrar_outliers_intercuartil:
            df_vari['IQR'] = df_vari[a_vary[i][2]] - df_vari[a_vary[i][0]]
            df_vari['cota_inf'] = df_vari[a_vary[i][0]] - 1.5 * df_vari['IQR']
            df_vari['cota_sup'] = df_vari[a_vary[i][2]] + 1.5 * df_vari['IQR']
            for row in zip(df_vari['MH_inicio'], df_vari['Servicio_Sentido'],
                           df_vari['cota_inf'], df_vari['cota_sup']):
                select1 = ((df_fv[i]['MH_inicio'] == row[0]) &
                           (df_fv[i]['Servicio_Sentido'] == row[1]))
                select2 = ((df_fv[i][a_vgrafricar[i]] >= row[2]) &
                           (df_fv[i][a_vgrafricar[i]] <= row[3]))
                df_fv[i] = df_fv[i].loc[((select1 & select2) | (~select1))]

            df_vari = df_fv[i].groupby(by=columnas_groupby).describe().reset_index()
            df_vari.columns = ['_'.join(col).rstrip('_') for col in df_vari.columns.values]
            # filtrar MH con menos de 2 datos
            df_vari = df_vari.loc[df_vari[f'{a_vgrafricar[i]}_count'] >= minimos_datos_por_mh]

        # pasar MH a datetime en una nueva columna
        df_vari['Media Hora'] = df_vari['MH_inicio'].map(dict_mh_date_str)
        df_var.append(df_vari.copy())

    # plotear
    contador = 0
    df_cero = pd.DataFrame(mh_del_dia, columns=['Media Hora'])
    df_cero['Cero'] = 0
    hra_min = min(df_var[0]['Media Hora'].min(), df_var[1]['Media Hora'].min())
    hra_max = max(df_var[0]['Media Hora'].max(), df_var[1]['Media Hora'].max())
    df_cero = df_cero.loc[df_cero['Media Hora'] >= hra_min]
    df_cero = df_cero.loc[df_cero['Media Hora'] <= hra_max]
    nombre_cero = '0'
    if filtrar_outliers_intercuartil:
        nombre_cero = 's0'

    nombre_ = ''
    if nombre:
        nombre_ = nombre.replace('-', '_')

    # max_data_count = max(df_var[0][a_vary[0][3]].max(), df_var[1][a_vary[1][3]].max())
    max_data_vary = df_var[0][a_vary[0][2]].max() * 1.5 + 0.01
    max_data_vary2 = df_var[1][a_vary[1][2]].max() + 1

    # iterar servicios
    for ss in df_var[0]['Servicio_Sentido'].unique():
        if ss not in df_var[1]['Servicio_Sentido'].unique():
            logger.warning(f'{ss} tiene datos de {variable_graficar} pero '
                           f'no de {variable_graficar_2}')

        logger.info(f'Graficando {variable_graficar} y {variable_graficar_2} {ss}')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # agregar un 0 para forzar mostrar el origen 0, 0
        fig.add_trace(go.Scatter(x=df_cero['Media Hora'].dt.time,
                                 y=df_cero['Cero'],
                                 name=nombre_cero,
                                 marker_color="white"))

        for i in [0, 1]:
            el_color = colores_2[contador % len(colores_2)][0]
            dfx = pd.merge(df_cero[['Media Hora']],
                           df_var[i].loc[df_var[i]['Servicio_Sentido'] == ss],
                           how='left',
                           on='Media Hora')

            if incluir_p75y25:
                fig.add_trace(
                    go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][2]],
                               name=f'percentil75',
                               mode='lines',
                               connectgaps=True,
                               opacity=opacity_percentiles,
                               line_color=el_color),
                    secondary_y=bool(i))

            fig.add_trace(
                go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][1]],
                           name=f'Mediana {dict_leyenda[a_vgrafricar[i]]}',
                           mode='lines+markers',
                           connectgaps=True,
                           marker=dict(size=marker_size,
                                       symbol=marcadores[contador % len(marcadores)]),
                           line_color=el_color),
                secondary_y=bool(i))

            if incluir_p75y25:
                fig.add_trace(
                    go.Scatter(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][0]],
                               name=f'percentil25',
                               mode='lines',
                               connectgaps=True,
                               opacity=opacity_percentiles,
                               line_color=el_color),
                    secondary_y=bool(i))

            # Agregar bar plot abajo con numero de datos
            fig.add_trace(go.Bar(x=dfx['Media Hora'].dt.time, y=dfx[a_vary[i][3]],
                                 marker=dict(color=el_color),
                                 opacity=opacity_barras,
                                 name=f'Nro Datos {dict_leyenda[a_vgrafricar[i]]}',
                                 width=[ancho_barras * 0.6] * len(dfx['Media Hora'].index)),
                          secondary_y=True)

            contador += 1

        # Formatear eje y secundario
        fig.update_yaxes(title_text="",
                         range=[0, max_data_vary2],
                         secondary_y=True)

        # Set x-axis title
        fig.update_xaxes(title_text="Nro Datos - Media hora despacho",
                         showticklabels=True,
                         type='category',
                         tickangle=270
                         )

        texto_titulo = (f"{dict_leyenda[a_vgrafricar[0]]} y "
                        f"{dict_leyenda[a_vgrafricar[1]]} por expedición {ss}"
                        f" (Dias {tipo_dia} de semana {nombre})")
        fig.update_yaxes(title_text="",
                         range=[0, max_data_vary],
                         gridcolor=colorLineas_ejeYppal,
                         secondary_y=False)
        if variable_graficar == 'delta_soc':
            fig.update_yaxes(tickformat=".1%", secondary_y=False)

        # Add figure title
        fig.update_layout(title=go.layout.Title(
            text=texto_titulo,
            font=dict(size=20, color='#000000')),
            font=dict(size=14, color='#000000'),
            xaxis_tickformat='%H:%M'
        )

        if filtrar_outliers_intercuartil:
            fig.write_html(f'graf_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'graf_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)
        else:
            fig.write_html(f'grafico_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.html',
                           config={'scrollZoom': True, 'displayModeBar': True})
            fig.write_image(f'grafico_{ss}_{variable_graficar}_{variable_graficar_2}_{tipo_dia}_{nombre_}.png',
                            width=1600, height=800)

    os.chdir('..')


def graficar_semana(dia_ini_, mes_, anno_, sem_especial=[], tipo_dia_='Laboral'):
    global primera_semana
    global ultima_semana
    global df_final
    df_final = []
    primera_semana = ''
    ultima_semana = ''

    g_pipeline(dia_ini_, mes_, anno_, sem_especial=sem_especial, tipo_dia=tipo_dia_)

    df_final = pd.concat(df_final)
    if df_final.empty:
        logger.warning(f'Esta semana no tiene data tracktec en su(s) dia(s) tipo {tipo_dia_}')
        return

    sem_primera = primera_semana.replace('semana_', '')
    carpeta_guardar_graficos = f'graficos_{sem_primera}'

    if not os.path.isdir(carpeta_guardar_graficos):
        logger.info(f'Creando carpeta {carpeta_guardar_graficos}')
        os.mkdir(carpeta_guardar_graficos)
    else:
        logger.warning(f'Reescribiendo sobre carpeta {carpeta_guardar_graficos}')

    os.chdir(carpeta_guardar_graficos)
    df_final.to_excel(f'data_{tipo_dia_}_{carpeta_guardar_graficos}.xlsx', index=False)
    df_final.to_parquet(f'data_{tipo_dia_}_{carpeta_guardar_graficos}.parquet', compression='gzip')

    sem_primera = sem_primera.replace('_', '-')
    logger.info(f'Graficando {tipo_dia_} {sem_primera}')

    graficar('tiempo_viaje', tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar('delta_soc', tipo_dia=tipo_dia_, nombre=sem_primera)
    # graficar('delta_Pcon', tipo_dia=tipo_dia_, nombre=sem_primera)
    # graficar('delta_Pgen', tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar_boxplot('delta_soc', tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar_soc_tv(tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar_potencias_2('delta_Pcon', 'delta_Pgen', tipo_dia=tipo_dia_, nombre=sem_primera)
    os.chdir('..')


def graficar_varias_semanas(tipo_dia_='Laboral'):
    global primera_semana
    global ultima_semana
    global df_final
    df_final = []
    primera_semana = ''
    ultima_semana = ''

    g_pipeline(7, 9, 2020, sem_especial=[], tipo_dia=tipo_dia_)
    g_pipeline(14, 9, 2020, sem_especial=[1, 2, 3, 6, 7], tipo_dia=tipo_dia_)
    g_pipeline(21, 9, 2020, sem_especial=[], tipo_dia=tipo_dia_)
    g_pipeline(28, 9, 2020, sem_especial=[], tipo_dia=tipo_dia_)
    g_pipeline(5, 10, 2020, sem_especial=[], tipo_dia=tipo_dia_)

    df_final = pd.concat(df_final)
    if df_final.empty:
        logger.warning(f'Esta semana no tiene data tracktec en su(s) dia(s) tipo {tipo_dia_}')
        return

    sem_primera = primera_semana.replace('semana_', '')
    sem_ultima = ultima_semana.replace('semana_', '')
    nombre_ = sem_primera
    if sem_primera == sem_ultima:
        carpeta_guardar_graficos = f'graficos_{sem_primera}'
    else:
        carpeta_guardar_graficos = f'graficos_{sem_primera}_{sem_ultima}'
        nombre_ = f'{sem_primera}_{sem_ultima}'

    if not os.path.isdir(carpeta_guardar_graficos):
        logger.info(f'Creando carpeta {carpeta_guardar_graficos}')
        os.mkdir(carpeta_guardar_graficos)
    else:
        logger.warning(f'Reescribiendo sobre carpeta {carpeta_guardar_graficos}')

    os.chdir(carpeta_guardar_graficos)
    df_final.to_excel(f'data_{tipo_dia_}_{carpeta_guardar_graficos}.xlsx', index=False)
    df_final.to_parquet(f'data_{tipo_dia_}_{carpeta_guardar_graficos}.parquet', compression='gzip')

    nombre_ = nombre_.replace('_', '-')
    logger.info(f'Graficando {tipo_dia_} {nombre_}')

    graficar('tiempo_viaje', tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar('delta_soc', tipo_dia=tipo_dia_, nombre=sem_primera)
    # graficar('delta_Pcon', tipo_dia=tipo_dia_, nombre=sem_primera)
    # graficar('delta_Pgen', tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar_boxplot('delta_soc', tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar_soc_tv(tipo_dia=tipo_dia_, nombre=sem_primera)
    graficar_potencias_2('delta_Pcon', 'delta_Pgen', tipo_dia=tipo_dia_, nombre=sem_primera)
    os.chdir('..')


def main():
    logger = mantener_log()
    # tipo_dia_interes puede ser 'Laboral' o 'Sabado' o 'Domingo'
    for tipo_dia_interes in ['Laboral', 'Sabado', 'Domingo']:
        # graficar_semana(7, 9, 2020, sem_especial=[], tipo_dia_=tipo_dia_interes)
        # graficar_semana(14, 9, 2020, sem_especial=[1, 2, 3, 6, 7], tipo_dia_=tipo_dia_interes)
        # graficar_semana(21, 9, 2020, sem_especial=[], tipo_dia_=tipo_dia_interes)
        # graficar_semana(28, 9, 2020, sem_especial=[], tipo_dia_=tipo_dia_interes)
        # graficar_semana(5, 10, 2020, sem_especial=[], tipo_dia_=tipo_dia_interes)

        # revisar que graficar_varias_semanas tenga estas mismas semanas
        graficar_varias_semanas(tipo_dia_=tipo_dia_interes)
        logger.info(f'Listo dias tipo {tipo_dia_interes}')

    logger.info('Listo todo')


if __name__ == '__main__':
    main()
