#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import logging
from datetime import timedelta
from geopy import distance
import ftplib
import os
global logger
global file_format


os.chdir('Cruce_Sonda_Ttec')


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


# mantener un log y guardarlo
logger = mantener_log()
file_handler = logging.FileHandler('log.log')
# no se escriben en archivo los debug, solo info hasta critical
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

def descargar_GPS_ftp(fecha__):
    direccion_resumen = ('Bruno/Data_PerdidaTransmision/' + fecha__[:4] +
                         '/' + fecha__[5:7] + '/' + fecha__[-2:])

    # filename = f'Cruce_196resumen_data_{fecha__}_revisado.xlsx'
    filename_gps = f'data_{fecha__}.parquet'

    hostname = '192.168.11.101'
    username = 'bruno'
    passw = 'manzana'
    max_reintentos = 20

    for tt in range(1, max_reintentos + 1):
        logger.info('Bajando GPS dia %s: intento número %d' % (fecha__, tt,))
        try:
            ftp = ftplib.FTP(hostname)
            ftp.login(username, passw)
            ftp.cwd(direccion_resumen)
            ftp.retrbinary("RETR " + filename_gps, open(filename_gps, 'wb').write)
            ftp.quit()
            break
        except (TimeoutError, ConnectionResetError):
            sleep(30 * tt)
    else:
        logger.warning(f'No se pudo conectar al servidor en '
                       f'{max_reintentos} intentos.')
        raise TimeoutError
    logger.info('Descargado.')


def descargar_semana_GPS_ftp(fechas, reemplazar=True):
    for fecha_ in fechas:
        filename_ = f'data_{fecha_}.parquet'
        if reemplazar or not os.path.isfile(filename_):
            descargar_GPS_ftp(fecha_)
        else:
            logger.info(f"No se va a reemplazar data FTP de fecha {fecha_}")
    logger.info('Todo descargado.')


def crear_directorio_ftp(currentDir, ftpx):
    if currentDir != "":
        try:
            ftpx.cwd(currentDir)
            return None
        except ftplib.error_perm:
            crear_directorio_ftp("/".join(currentDir.split("/")[:-1]), ftpx)
            ftpx.mkd(currentDir)
            ftpx.cwd(currentDir)
            return None


hostname = '192.168.11.101'
username = 'bruno'
passw = 'manzana'
max_reintentos = 20
archivos_subir = os.listdir('.')
archivos_subir = [x for x in os.listdir('.') if x[-8:] == '.parquet']


direccion_base = '/home/apple/Documentos/soc_por_expedicion'
carpetas = [x for x in os.listdir(direccion_base) if x[:7] == 'semana_']
logger.info(f'Carpetas por semana: {carpetas}')

for carpeta in carpetas:
    direccion = f'{direccion_base}/{carpeta}'
    archivos = [x for x in os.listdir(direccion) if x[:10] == 'data_Ttec_']
    fechas = [x.replace('data_Ttec_', '') for x in archivos]
    fechas = [x.replace('.parquet', '') for x in fechas]

    saltar_fechas = []
    for fecha in fechas:
        archivo_st = f'data_ST_{fecha}.parquet'
        if os.path.isfile(archivo_st):
            saltar_fechas.append(fecha)

    fechas = [x for x in fechas if x not in saltar_fechas]
    if not fechas:
        logger.info(f'Carpeta {carpeta} lista')
        continue

    logger.info(f'La carpeta {carpeta} tiene las fechas: {fechas}')
    descargar_semana_GPS_ftp(fechas, reemplazar=False)

    for fecha in fechas:
        logger.info(f'Cruzando data fecha {fecha}')

        df_T = pd.read_parquet(f'{direccion}/data_Ttec_{fecha}.parquet')
        df_res = pd.read_parquet(f'{direccion}/data_196rE_{fecha}.parquet')
        df_S = pd.read_parquet(f'data_{fecha}.parquet')

        df_T = df_T.loc[df_T['patente'].isin(df_res['patente'].unique())]
        df_T = df_T.loc[~df_T['valor_soc'].isna()]

        df_S = df_S.loc[df_S['Ruta']!=0]
        # SS196 incluye sentido, se saca con str[:-1]
        df_S = df_S.loc[df_S['SS196'].str[:-1].isin(df_res['Servicio'].unique())]
        df_S = df_S.loc[df_S['Ruta'].isin(df_res['Indice_mensual'].unique())]
        df_S = df_S.loc[df_S['patente'].isin(df_T['patente'].unique())]

        df_T.sort_values(by=['fecha_hora_evento'], inplace=True)
        df_S.sort_values(by=['timestamp'], inplace=True)
        logger.info(f'N Datos Telemetria Tracktec {len(df_T.index)}')
        logger.info(f'N Datos GPS Webservice Sonda {len(df_S.index)}')
        df_ST = pd.merge_asof(df_S, df_T,
                              left_on='timestamp',
                              right_on='fecha_hora_evento',
                              left_by='patente', right_by='patente',
                              suffixes=['', '_Ttec'],
                              tolerance=timedelta(minutes=1),
                              direction='nearest')
        df_ST.sort_values(by=['patente', 'timestamp'], inplace=True)
        df_ST = df_ST.loc[~df_ST['valor_soc'].isna()]
        if len(df_ST.index) > 10:
            logger.info(f'N Datos Cruce Sonda Tracktec {len(df_ST.index)}')

            # solo quedarse con el pulso mas cercano al dato tracktec

            df_ST['dT_merge'] = abs((df_ST['timestamp'] -
                                     df_ST['fecha_hora_evento']) / pd.Timedelta(seconds=1))
            df_ST.sort_values(by=['patente', 'fecha_hora_evento', 'dT_merge'], inplace=True)
            df_ST.drop_duplicates(subset=['patente', 'fecha_hora_evento'], keep='first', inplace=True)
            df_ST.sort_values(by=['patente', 'timestamp'], inplace=True)

            logger.info(f'N Datos Cruce procesados {len(df_ST.index)}')
            if False:  # En caso que se quiera data Sonda con y sin SOC
                # Ahora esta data de SOC por pulso agregarla como columna nueva a data sonda
                df_S = df_S.loc[df_S['Ruta'].isin(df_ST['Ruta'].unique())]

                logger.info(f'N Datos Sonda Utiles {len(df_S.index)}')
                df_S.reset_index(inplace=True)
                df_S = pd.merge(df_S, df_ST[['patente', 'timestamp', 'valor_soc', 'fecha_hora_evento']],
                                how='left', on=['patente', 'timestamp'], sort=True, suffixes=['', ''])
                df_S.set_index('id', inplace=True)
                logger.info(f'N Datos Sonda Finales {len(df_S.index)}')

                df_S.to_parquet(f'data_STf_{fecha}.parquet', compression='gzip')
            else:
                df_ST.to_parquet(f'data_ST_{fecha}.parquet', compression='gzip')

        else:
            logger.info(f'No hay datos cruzados entre ambas tablas')

        os.remove(f'data_{fecha}.parquet')


logger.info('Listo cruce, moviendo archivos al ftp..')

for arch in archivos_subir:
    direccion_archivos = arch.replace('.parquet', '').replace('data_ST_','')
    direccion_archivos = direccion_archivos.replace('_', '/')

    for tt in range(1, max_reintentos + 1):
        logger.info(f'Subiendo {arch} a FTP: intento número {tt}')
        try:
            ftp = ftplib.FTP(hostname)
            ftp.login(username, passw)

            crear_directorio_ftp(f'/Bruno/Data_electricos/{direccion_archivos}', ftp)
            ftp.cwd(f'/Bruno/Data_electricos/{direccion_archivos}')
            fp = open(f'{arch}', 'rb')
            ftp.storbinary(f'STOR {arch}', fp, 1024)
            fp.close()

            ftp.quit()
            break
        except (TimeoutError, ConnectionResetError):
            sleep(30 * tt)
    else:
        logger.info(f'ERROR: No se pudo conectar al servidor en {max_reintentos} intentos.')
        raise TimeoutError

logger.info(f'Ahora esta todo en el FTP')

