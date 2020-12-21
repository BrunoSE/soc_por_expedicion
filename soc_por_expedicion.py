import MySQLdb
import pandas as pd
import logging
import ftplib
import os
from datetime import timedelta
from geopy import distance
from time import sleep
from sys import platform

global logger
global file_format

if platform.startswith('win'):
    ip_bd_edu = "26.2.206.141"
else:
    ip_bd_edu = "192.168.11.150"


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


def consultar_soc_id(id_evento):
    db1 = MySQLdb.connect(host=ip_bd_edu,
                          user="brunom",
                          passwd="Manzana",
                          db="tracktec")

    cur1 = db1.cursor()

    cur1.execute("SELECT valor FROM telemetria_" +
                 f" WHERE evento_id = {id_evento} AND nombre = 'SOC' ;")

    datos = [row for row in cur1.fetchall() if row[0] is not None]
    cur1.close()
    db1.close()
    return datos


def consultar_numero_transmisiones_por_semana(fecha_inicial, fecha_final):
    db1 = MySQLdb.connect(host=ip_bd_edu,
                          user="brunom",
                          passwd="Manzana",
                          db="tracktec")

    cur1 = db1.cursor()
    cur1.execute("SELECT fecha_evento, count(fecha_evento) FROM tracktec.eventos " +
                 f"WHERE fecha_evento >= '{fecha_inicial}' AND fecha_evento <= '{fecha_final}'" +
                 " AND hora_evento IS NOT NULL AND bus_tipo = 'Electric' GROUP BY fecha_evento")

    datos = [row for row in cur1.fetchall() if row[0] is not None]
    cur1.close()
    db1.close()
    return datos


def consultar_transmisiones_por_semana(fecha_inicial, fecha_final):
    db1 = MySQLdb.connect(host=ip_bd_edu,
                          user="brunom",
                          passwd="Manzana",
                          db="tracktec")

    cur1 = db1.cursor()
    cur1.execute("SELECT * FROM tracktec.eventos " +
                 f"WHERE fecha_evento >= '{fecha_inicial}' AND fecha_evento <= '{fecha_final}'" +
                 " AND hora_evento IS NOT NULL AND bus_tipo = 'Electric' " +
                 "LIMIT 10")

    datos = [row for row in cur1.fetchall() if row[0] is not None]
    df_ = pd.DataFrame(datos, columns=[i[0] for i in cur1.description])

    cur1.close()
    db1.close()
    return df_


def procesar_datos_consulta(cursor):
    datos = [row for row in cursor.fetchall() if row[0] is not None]
    df_ = pd.DataFrame(datos, columns=[i[0] for i in cursor.description])
    df_.set_index('id', inplace=True)
    for columna in ['latitud', 'longitud', 'valor_soc', 'valor_ptg', 'valor_ptc']:
        if columna in df_.columns:
            try:
                df_[columna] = pd.to_numeric(df_[columna])
            except ValueError:
                logger.exception(f'Error en columna {columna}')
        else:
            logger.warning(f'Columna {columna} no está en estos datos')

    df_['fecha_hora_consulta'] = pd.to_datetime(df_['fecha_hora_consulta'], errors='raise',
                                                format="%Y-%m-%d %H:%M:%S")
    df_['fecha_evento'] = pd.to_datetime(df_['fecha_evento'], errors='raise',
                                         format="%Y-%m-%d")
    df_['fecha_hora_evento'] = df_['fecha_evento'] + df_['hora_evento']

    return df_


def consultar_soc_ttec(fecha_dia):
    db1 = MySQLdb.connect(host=ip_bd_edu,
                          user="brunom",
                          passwd="Manzana",
                          db="tracktec")

    cur1 = db1.cursor()

    cur1.execute(
                 f"""
                 SELECT * FROM tracktec.eventos as te1 JOIN 
                 (SELECT evento_id as evento_id_soc, nombre as nombre_soc, 
                 valor as valor_soc FROM tracktec.telemetria_ 
                 WHERE (nombre = 'SOC' and 
                        valor REGEXP '^[\\-]?[0-9]+\\.?[0-9]*$')) as t_soc 
                 ON te1.id=t_soc.evento_id_soc 
                 WHERE fecha_evento = '{fecha_dia}'
                 AND hora_evento IS NOT NULL AND bus_tipo = 'Electric' 
                 AND PATENTE IS NOT NULL AND NOT (patente REGEXP '^[0-9]+')
                 ORDER BY patente;
                 """
                 )

    df__ = procesar_datos_consulta(cur1)

    cur1.close()
    db1.close()

    return df__


def consultar_transmisiones_tracktec_por_dia(fecha_dia):
    db1 = MySQLdb.connect(host=ip_bd_edu,
                          user="brunom",
                          passwd="Manzana",
                          db="tracktec")

    cur1 = db1.cursor()

    cur1.execute(
                 f"""
                 select * from
                 (
                     select * from 
                     (
                         select * from
                         (
                             SELECT * FROM tracktec.eventos 
                             as te1 left JOIN 
                                 (SELECT evento_id as evento_id_soc, nombre as nombre_soc, 
                                 valor as valor_soc FROM tracktec.telemetria_ 
                                 WHERE (nombre = 'SOC' and 
                                        valor REGEXP '^[\\-]?[0-9]+\\.?[0-9]*$')) as t_soc 
                                 ON te1.id=t_soc.evento_id_soc 
                                 WHERE fecha_evento = '{fecha_dia}'
                                 AND hora_evento IS NOT NULL AND bus_tipo = 'Electric' 
                                 AND PATENTE IS NOT NULL AND NOT (patente REGEXP '^[0-9]+')
                         ) as te2 left join
                             (SELECT evento_id as evento_id_ptg, nombre as nombre_ptg, 
                             valor as valor_ptg FROM tracktec.telemetria_ 
                             WHERE (nombre = 'Potencia Total Generada' and 
                                    valor REGEXP '^[\\-]?[0-9]+\\.?[0-9]*$')) as t_ptg 
                             ON te2.id=t_ptg.evento_id_ptg 
                             WHERE fecha_evento = '{fecha_dia}'
                             AND hora_evento IS NOT NULL AND bus_tipo = 'Electric' 
                             AND PATENTE IS NOT NULL AND NOT (patente REGEXP '^[0-9]+')
                     ) as te3 left join 
                         (SELECT evento_id as evento_id_ptc, nombre as nombre_ptc, 
                         valor as valor_ptc FROM tracktec.telemetria_ 
                         WHERE (nombre = 'Potencia Total Consumida' and 
                                valor REGEXP '^[\\-]?[0-9]+\\.?[0-9]*$')) as t_ptc 
                         ON te3.id=t_ptc.evento_id_ptc 
                         WHERE fecha_evento = '{fecha_dia}'
                         AND hora_evento IS NOT NULL AND bus_tipo = 'Electric' 
                         AND PATENTE IS NOT NULL AND NOT (patente REGEXP '^[0-9]+')
                 ) as te_final
                 where 
                 valor_soc IS NOT NULL OR
                 valor_ptg IS NOT NULL OR
                 valor_ptc IS NOT NULL
                 order by patente;
                 """
                 )

    df__ = procesar_datos_consulta(cur1)

    cur1.close()
    db1.close()

    return df__


def descargar_semana_ttec(fechas, reemplazar=True):
    for fecha_ in fechas:
        if reemplazar or not os.path.isfile(f'data_Ttec_{fecha_}.parquet'):
            fecha__ = fecha_.replace('_', '-')
            logger.info(f"Descargando data Tracktec para fecha {fecha_}")
            dfx = consultar_transmisiones_tracktec_por_dia(fecha__)
            dfx.to_parquet(f'data_Ttec_{fecha_}.parquet', compression='gzip')
        else:
            logger.info(f"No se va a reemplazar data Ttec de fecha {fecha_}")


def descargar_resumen_ftp(fecha__, descargar_data_gps=False):
    direccion_resumen = ('Bruno/Data_PerdidaTransmision/' + fecha__[:4] +
                         '/' + fecha__[5:7] + '/' + fecha__[-2:])

    filename = f'Cruce_196resumen_data_{fecha__}_revisado.xlsx'
    filename_gps = f'data_{fecha__}.parquet'

    hostname = '192.168.11.101'
    username = 'bruno'
    passw = 'manzana'
    max_reintentos = 20

    for tt in range(1, max_reintentos + 1):
        logger.info('Bajando resumen dia %s: intento número %d' % (fecha__, tt,))
        try:
            ftp = ftplib.FTP(hostname)
            ftp.login(username, passw)
            ftp.cwd(direccion_resumen)
            ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
            if descargar_data_gps:
                ftp.retrbinary("RETR " + filename_gps, open(filename_gps, 'wb').write)
            ftp.quit()
            break
        except (TimeoutError, ConnectionResetError):
            sleep(30 * tt)
    else:
        logger.warning(f'No se pudo conectar al servidor en '
                       f'{max_reintentos} intentos.')
        raise TimeoutError


def descargar_semana_ftp(fechas, reemplazar=True, descargar_data_gps_=False):
    for fecha_ in fechas:
        filename_ = f'Cruce_196resumen_data_{fecha_}_revisado.xlsx'
        if descargar_data_gps_ or (reemplazar or not os.path.isfile(filename_)):
            descargar_resumen_ftp(fecha_, descargar_data_gps_)
        else:
            logger.info(f"No se va a reemplazar data FTP de fecha {fecha_}")


def distancia_wgs84(lat1: float, lon1: float, lat2: float, lon2: float):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None

    return 1000 * distance.distance((lat1, lon1), (lat2, lon2)).km


def mezclar_data(fecha):
    servicios_de_interes = ['F41', 'F46', 'F48', 'F63c', 'F67e', 'F83c',
                            'F69', 'F73', 'F81']

    df196r = pd.read_excel(f'Cruce_196resumen_data_{fecha}_revisado.xlsx')
    logger.info(f"Expediciones iniciales en resumen diario: {len(df196r.index)}")
    df196r = df196r.loc[df196r['Servicio'].isin(servicios_de_interes)]
    df196r = df196r.loc[~(df196r['hora_inicio'].isna())]
    # df196r = df196r.loc[(df196r['Operativo'] == 'C')]
    # df196r = df196r.loc[(df196r['Cumple_Triada_Revisada'] == 1)]
    # df196r = df196r.loc[(df196r['Cumple_TVE_TV '] == 'C')]
    # df196r = df196r.loc[(df196r['Pulsos_por_min'] >= 1.75)]
    # df196r = df196r.loc[(df196r['pctje_pulsos_FDR'] < 0.3)]

    # df196r['pctje_dist_recorrida'] = df196r['distancia_recorrida'] / df196r['dist_Ruta']
    # df196r = df196r.loc[df196r.pctje_dist_recorrida > 0.85]
    # df196r = df196r.loc[df196r.pctje_dist_recorrida < 1.15]

    # filtros adicionales que podria incluir:
    # tiempo con perdida transmision menor a 5 min
    # distancia con perdida de transmision menor a 1 km
    # no mas de 3 pulsos fuera de ruta

    # luego de procesar:
    # filtrar por valores delta_ soc potencia > 0 (positivos no-nulos)
    # distancia dato Tracktec a registro gps Sonda asignado < 1km

    df = pd.read_parquet(f'data_Ttec_{fecha}.parquet')
    # para que todas las columnas vengan renombradas
    columnas_originales = df.columns
    df.columns = columnas_originales + '_Ttec_ini'

    # llevar log de info relevante
    logger.info(f"Largo data tracktec: {len(df.index)}")
    logger.info(f"Número total de expediciones con SS relevantes: {len(df196r.index)}")
    logger.info(f"Número de expediciones por SS:\n"
                f"{repr(df196r['Servicio_Sentido'].value_counts())}")

    # df = df.loc[df['patente'] == 'PFVC-40']
    # logger.info(f"Largo data tracktec con la ppu: {len(df.index)}")
    df.sort_values(by=['fecha_hora_evento_Ttec_ini'], inplace=True)
    df196r.sort_values(by=['hora_inicio'], inplace=True)

    df196r_e = pd.merge_asof(df196r, df,
                             left_on='hora_inicio',
                             right_on='fecha_hora_evento_Ttec_ini',
                             left_by='PPU', right_by='patente_Ttec_ini',
                             suffixes=['', '_Ttec_ini2'],
                             tolerance=timedelta(minutes=1),
                             direction='nearest')

    df196r_e.sort_values(by=['hora_fin'], inplace=True)
    df.columns = columnas_originales + '_Ttec_fin'

    df196r_ef = pd.merge_asof(df196r_e, df,
                              left_on='hora_fin',
                              right_on='fecha_hora_evento_Ttec_fin',
                              left_by='PPU', right_by='patente_Ttec_fin',
                              suffixes=['', '_Ttec_fin2'],
                              tolerance=timedelta(minutes=1),
                              direction='nearest')

    # agregar primera y ultima posicion a mi resumen diario de gps
    df196r_ef['d_registros_ini'] = df196r_ef.apply(lambda x: distancia_wgs84(x['latitud_Ttec_ini'],
                                                                             x['longitud_Ttec_ini'],
                                                                             x['lat_ini'],
                                                                             x['lon_ini']), axis=1)

    df196r_ef['d_registros_fin'] = df196r_ef.apply(lambda x: distancia_wgs84(x['latitud_Ttec_fin'],
                                                                             x['longitud_Ttec_fin'],
                                                                             x['lat_fin'],
                                                                             x['lon_fin']), axis=1)

    df196r_ef['delta_soc'] = df196r_ef['valor_soc_Ttec_ini'] - df196r_ef['valor_soc_Ttec_fin']
    df196r_ef['delta_Pcon'] = df196r_ef['valor_ptc_Ttec_fin'] - df196r_ef['valor_ptc_Ttec_ini']
    df196r_ef['delta_Pgen'] = df196r_ef['valor_ptg_Ttec_ini'] - df196r_ef['valor_ptg_Ttec_fin']
    df196r_ef.sort_values(by=['PPU', 'hora_inicio'], inplace=True)
    df196r_ef['Intervalo'] = pd.to_datetime(df196r_ef['Intervalo'], errors='raise',
                                            format="%H:%M:%S")

    df196r_ef.to_parquet(f'data_196rE_{fecha}.parquet', compression='gzip')
    return None


def pipeline(dia_ini, mes, anno, replace_data_ttec=False, replace_resumen=False, sem_especial=[]):
    # dia_ini tiene que ser un día lunes
    # Sacar fechas de interes a partir de lunes inicio de semana
    fecha_dia_ini = pd.to_datetime(f'{dia_ini}-{mes}-{anno}', dayfirst=True).date()
    dia_de_la_semana = fecha_dia_ini.isoweekday()
    if dia_de_la_semana != 1:
        logger.error(f"Primer día no es lunes y se quiere ocupar parámetro sem_especial, "
                     f"numero dia_ini: {dia_de_la_semana}")
        exit()

    fechas_de_interes = []
    if not sem_especial:
        for i in range(0, 7):
            fechas_de_interes.append(fecha_dia_ini + pd.Timedelta(days=i))
    else:
        # se buscan días de la semana entre fecha inicio y el domingo
        if len(sem_especial) != len(set(sem_especial)):
            logger.error(f"Semana especial no debe repetir números: {sem_especial}")
            exit()
        for i in sem_especial:
            if 0 < i < 8:
                fechas_de_interes.append(fecha_dia_ini + pd.Timedelta(days=(i - 1)))
            else:
                logger.error(f"Semana especial debe ser lista con números 1 al 7: {sem_especial}")
                exit()
    fechas_de_interes = [x.strftime('%Y-%m-%d') for x in fechas_de_interes]

    logger.info(f'Semana de interes: {fechas_de_interes}')

    no_existia_semana = False
    nombre_semana = f"semana_{fechas_de_interes[0].replace('-', '_')}"

    # buscar si ya existia carpeta
    if not os.path.isdir(nombre_semana):
        logger.info(f'Creando carpeta {nombre_semana}')
        os.mkdir(nombre_semana)
        no_existia_semana = True
    else:
        logger.info(f'Se encontró carpeta {nombre_semana}')
        if replace_resumen:
            logger.info("Como replace_resumen=True se van a reemplazar los archivos resumen")
        if replace_data_ttec:
            logger.info("Como replace_data_ttec=True se van a reemplazar archivos parquet")
        elif not replace_resumen:
            logger.info("No se van a reemplazar archivos")

    os.chdir(nombre_semana)

    # Crear variable que escribe en log file de este dia
    file_handler = logging.FileHandler(f'{nombre_semana}.log')

    # no deja pasar los debug, solo info hasta critical
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    fechas_de_interes = [x.replace('-', '_') for x in fechas_de_interes]

    if no_existia_semana:
        replace_data_ttec = True
        replace_resumen = True

    logger.info('Consultando servidor mysql por datos tracktec')
    descargar_semana_ttec(fechas_de_interes, replace_data_ttec)

    logger.info('Descargando archivos de resumen del FTP')
    descargar_semana_ftp(fechas_de_interes, replace_resumen)

    for fi in fechas_de_interes:
        logger.info(f'Concatenando y mezclando data de fecha {fi}')
        mezclar_data(fi)

    logger.info('Listo todo para esta semana')
    os.chdir('..')


if __name__ == '__main__':
    logger = mantener_log()

    reemplazar_data_ttec = False
    reemplazar_resumen = True
    # pipeline(7, 9, 2020, reemplazar_data_ttec, reemplazar_resumen)
    # pipeline(14, 9, 2020, reemplazar_data_ttec, reemplazar_resumen, sem_especial=[1, 2, 3, 6, 7])
    # pipeline(21, 9, 2020, reemplazar_data_ttec, reemplazar_resumen)
    # pipeline(28, 9, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(2, 11, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(9, 11, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(16, 11, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(23, 11, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(30, 11, 2020, reemplazar_data_ttec, reemplazar_resumen)
    pipeline(7, 12, 2020, reemplazar_data_ttec, reemplazar_resumen)
    logger.info('Listo todo')
