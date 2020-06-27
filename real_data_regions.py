from constants import ROOT_DIR
import pandas as pd
import numpy as np

# N.B. Trento and Bolzano are not regions, those populations data are therefore taken from Wikipedia
regions_populations = {'Abruzzo': 1311580,
                       'Basilicata': 562869,
                       'P.A. Bolzano': 106951,
                       'Calabria': 1947131,
                       'Campania': 5801692,
                       'Emilia-Romagna': 4459477,
                       'Friuli Venezia Giulia': 1215220,
                       'Lazio': 5879082,
                       'Liguria': 1550640,
                       'Lombardia': 10060574,
                       'Marche': 1525271,
                       'Molise': 305617,
                       'Piemonte': 4356406,
                       'Puglia': 4029053,
                       'Sardegna': 1639591,
                       'Sicilia': 4999891,
                       'Toscana': 3729641,
                       'P.A. Trento': 541380,
                       'Umbria': 882015,
                       'Valle d\'Aosta': 125666,
                       'Veneto': 4905854}

regions_to_fit = pd.read_csv(ROOT_DIR + '/real_data/regioni_to_fit.csv')
regions = regions_to_fit['denominazione_regione'].unique()

regions_dict = {}

for r in regions:
    infected = np.array(regions_to_fit[regions_to_fit['denominazione_regione'] == r]['totale_positivi'])
    recovered = np.array(regions_to_fit[regions_to_fit['denominazione_regione'] == r]['dimessi_guariti'])
    deaths = np.array(regions_to_fit[regions_to_fit['denominazione_regione'] == r]['deceduti'])
    new_cases = np.array(regions_to_fit[regions_to_fit['denominazione_regione'] == r]['nuovi_positivi'])

    removed = recovered + deaths

    infected = infected / regions_populations[r]
    removed = removed / regions_populations[r]

    regions_dict[r] = [infected, removed, new_cases]






