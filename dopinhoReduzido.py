import pubchempy as pcp  # PubChemPy 1.0.4
from pymsfilereader import MSFileReader  # pymsfilereader 1.0.1
import pandas as pd  # pandas 1.5.3
import matplotlib.pyplot as plt  # matplotlib 3.9.2
from sklearn.metrics import r2_score  # scikit-learn 1.5.2
import os  # biblioteca padrão do Python
import numpy as np  # numpy 1.24.0
import skfuzzy as fuzz  # scikit-fuzzy 0.5.0
import timeit  # biblioteca padrão do Python
import textwrap  # biblioteca padrão do Python
from rdkit import Chem  # rdkit 2024.3.6
from rdkit.Chem import Draw  # rdkit 2024.3.6
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # matplotlib 3.9.2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # matplotlib 3.9.2
import pkg_resources  # biblioteca padrão do Python
#PYTHON VERSION 3.10.0

start_time = timeit.default_timer()

def buscar_informacoes_pubchem(cid):
    try:
        # Recupera o objeto Compound diretamente
        composto = pcp.Compound.from_cid(cid)

        if composto:
            # Extrai os atributos desejados com checagem de existência
            nome_iupac = composto.iupac_name if composto.iupac_name else 'N/A'
            nome_substancia = composto.synonyms[0] if composto.synonyms else 'N/A'
            mass = composto.exact_mass if composto.exact_mass else 'N/A'
            molecula = composto.molecular_formula if composto.molecular_formula else 'N/A'
            peso = composto.molecular_weight if composto.molecular_weight else 'N/A'
            sinonimos = composto.synonyms[:3] if composto.synonyms[:3] else 'N/A'
            structe = composto.isomeric_smiles if composto.isomeric_smiles else 'N/A'
            
            # Busca o número CAS (se existir) nos sinônimos
            cas = None
            for syn in composto.synonyms:
                if '-' in syn and syn.replace('-', '').isdigit():  # Verifica se o formato é numérico com hífen
                    cas = syn
                    break

            cas = cas if cas else 'N/A'
            
          
            return {
                'nome_iupac': nome_iupac,
                'cas': cas,
                'nome_substancia': nome_substancia,
                'massa': mass,
                'molecula': molecula,
                'peso': peso,
                'sinonimos': sinonimos,
                'structe': structe
            }
        else:
            print(f"Nenhum resultado encontrado para CID {cid}")
            return {
                'nome_iupac': 'N/A',
                'cas': 'N/A',
                'nome_substancia': 'N/A',
                'massa': 'N/A',
                'molecula': 'N/A',
                'peso': 'N/A',
                'sinonimos': 'N/A',
                'structe': 'N/A'
            }

    except Exception as e:
        print(f"Erro ao buscar informações para CID {cid}: {e}")
        return {
             'nome_iupac': 'N/A',
             'cas': 'N/A',
             'nome_substancia': 'N/A',
             'massa': 'N/A',
             'molecula': 'N/A',
             'peso': 'N/A',
             'sinonimos': 'N/A',
             'structe': 'N/A'
        }

def obter_dados_e_processar(leitor, start_time, end_time, mass_range, scan_filter):
    dados = leitor.GetChroData(
        startTime=start_time,
        endTime=end_time,
        massRange1=mass_range,
        scanFilter=scan_filter,
        smoothingType=2,
        smoothingValue=7
    )
    return pd.DataFrame({'col1': dados[0][0], 'col2': dados[0][1]})

def fuzzy_classificacao(coef_determinacao):
    x_coef = np.arange(-12, 2, 0.1)
    x_classificacao = np.arange(0, 6, 1)

    coef_muito_baixo = fuzz.trapmf(x_coef, [-12, -12, -8, -4])
    coef_baixo = fuzz.trimf(x_coef, [-8, -4, -2])
    coef_medio = fuzz.trimf(x_coef, [-4, -2, 0])
    coef_alto = fuzz.trimf(x_coef, [-2, 0, 1])
    coef_muito_alto = fuzz.trimf(x_coef, [0, 1, 2])
    
    class_negativo = fuzz.trimf(x_classificacao, [0, 0, 1])
    class_muito_baixo = fuzz.trimf(x_classificacao, [1, 1, 2])
    class_baixo = fuzz.trimf(x_classificacao, [2, 2, 3])
    class_medio = fuzz.trimf(x_classificacao, [3, 3, 4])
    class_alto = fuzz.trimf(x_classificacao, [4, 4, 5])
    class_muito_alto = fuzz.trimf(x_classificacao, [5, 5, 6])
    
    grau_muito_baixo = fuzz.interp_membership(x_coef, coef_muito_baixo, coef_determinacao)
    grau_baixo = fuzz.interp_membership(x_coef, coef_baixo, coef_determinacao)
    grau_medio = fuzz.interp_membership(x_coef, coef_medio, coef_determinacao)
    grau_alto = fuzz.interp_membership(x_coef, coef_alto, coef_determinacao)
    grau_muito_alto = fuzz.interp_membership(x_coef, coef_muito_alto, coef_determinacao)
    
    regra1 = np.fmin(grau_muito_baixo, class_muito_baixo)
    regra2 = np.fmin(grau_baixo, class_baixo)
    regra3 = np.fmin(grau_medio, class_medio)
    regra4 = np.fmin(grau_alto, class_alto)
    regra5 = np.fmin(grau_muito_alto, class_muito_alto)
    
    agregacao = np.fmax(regra1, np.fmax(regra2, np.fmax(regra3, np.fmax(regra4, regra5))))
    
    if np.sum(agregacao) == 0:
        return 0
    
    classificacao = fuzz.defuzz(x_classificacao, agregacao, 'centroid')
    
    return classificacao

def plotar_e_comparar(arquivo1, arquivo2, arquivo3, arquivo4, nome1, nome2, nome3, nome4, nome_arquivo, cid, report_name):
    # Buscar informações adicionais no PubChem
    info_substancia = buscar_informacoes_pubchem(cid)
    
    min_length = min(len(arquivo1), len(arquivo2))
    min_lengthV = min(len(arquivo3), len(arquivo4))
    arquivo1 = arquivo1.iloc[:min_length]
    arquivo2 = arquivo2.iloc[:min_length]
    arquivo3 = arquivo3.iloc[:min_lengthV]
    arquivo4 = arquivo4.iloc[:min_lengthV]

    coef_determinacao = r2_score(arquivo1.iloc[:, 1], arquivo2.iloc[:, 1])
    coef_determinacao = float("{:.6f}".format(coef_determinacao))
    
    classificacao = fuzzy_classificacao(coef_determinacao)
    
    if classificacao == 0:
        result = "NEGATIVO"
    elif classificacao <= 1:
        result = "PRESUMIVEL MUITO BAIXO"
    elif classificacao <= 2:
        result = "PRESUMIVEL BAIXO"
    elif classificacao <= 3:
        result = "PRESUMIVEL MEDIO"
    elif classificacao <= 4:
        result = "PRESUMIVEL ALTO"
    elif classificacao <= 5:
        result = "PRESUMIVEL MUITO ALTO"
    else:
        result = "CQCL NÃO FORTIFICADO"

    arquivo = os.path.basename(nome_arquivo)
    #with open("output.txt", "a") as f:
    #    f.write(f"{arquivo} / {nome1} / {result} / 
    #            {coef_determinacao:.6f} IUPAC: {info_substancia['nome_iupac']}, 
    #            CAS: {info_substancia['cas']}, Mass:{info_substancia['massa']}, 
    #            Molecula: {info_substancia['molecula']}, 
    #            Peso: {info_substancia['peso']}, 
    #            Sinonimos: {info_substancia['sinonimos']}, 
    #            Structe: {info_substancia['structe']})")
        
    if not nome1.startswith("AIS"):
        if coef_determinacao > -12:
            if coef_determinacao != 0.0:
                if coef_determinacao != 1:
       
                    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

                    # Plotar os cromatogramas em ax[0]
                    ax[0].plot(arquivo1.iloc[:, 0], arquivo1.iloc[:, 1], label="AMOSTRA", linestyle='-', color='orange')
                    ax[0].plot(arquivo2.iloc[:, 0], arquivo2.iloc[:, 1], label="CQCL",  linestyle='--', color='purple')
                    ax[0].plot(arquivo3.iloc[:, 0], arquivo3.iloc[:, 1], label="CQN", linestyle='-', color='green')
                    ax[0].plot(arquivo4.iloc[:, 0], arquivo4.iloc[:, 1], label="CQCL_Reinj", linestyle='--', color='navy')

                    ax[0].set_xlabel('Tempo de Retenção')
                    ax[0].set_ylabel('Intensidade')
                    ax[0].legend()
                    ax[0].set_title(f"{report_name}")
                    
                    iupac_formatado = textwrap.fill(f"$\\bf{{IUPAC}}$: {info_substancia['nome_iupac']}", width=80)
                    sinonimos_formatado = textwrap.fill(f"$\\bf{{Sinônimos}}$: {info_substancia['sinonimos']}", width=80)

                    detalhes_substancia_parte1 = (
                        f"$\\bf{{Conclusão}}$: {result}\n"
                        f"$\\bf{{Amostra}}$: {arquivo}\n"
                        f"$\\bf{{Substância Lab}}$: {report_name}\n\n"
                        f"$\\bf{{Substância PubChem}}$: {info_substancia['nome_substancia']}\n"
                        f"{iupac_formatado}\n"
                        f"$\\bf{{CAS}}$: {info_substancia['cas']}\n"
                        f"$\\bf{{Mass}}$: {info_substancia['massa']}\n"
                        f"$\\bf{{Peso}}$: {info_substancia['peso']}\n"
                        f"$\\bf{{Molecula}}$: {info_substancia['molecula']}\n"
                        f"{sinonimos_formatado}\n"
                        f"$\\bf{{Estrutura}}$:\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                    )

                    ax[1].text(0.05, 1.0, detalhes_substancia_parte1, fontsize=11, va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'), linespacing=1.5)

                    # Criar molécula a partir do SMILES e adicionar imagem da molécula
                    mol = Chem.MolFromSmiles(info_substancia['structe'])
                    if mol:
                        # Gerar a imagem da molécula
                        mol_image = Draw.MolToImage(mol, size=(350, 350))

                        # Adicionar a imagem da molécula abaixo do texto "Estrutura"
                        imagebox = OffsetImage(mol_image, zoom=0.6)
                        ab = AnnotationBbox(imagebox, (0.3, 0.29), frameon=False, xycoords='axes fraction', boxcoords="axes fraction", pad=0)
                        ax[1].add_artist(ab)

                    # Configurar o gráfico principal (ax[1])
                    ax[1].axis('off')
                    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
                    plt.show()

        return coef_determinacao

def processar_arquivo_amostra(nome_arquivo, caminho_controle, caminho_controle_reinj, caminho_controle_negativo):
    MSp = "FTMS + p ESI Full ms [100.0000-670.0000]"
    MSn = "FTMS - p ESI Full ms [100.0000-670.0000]"

    amostra = MSFileReader(nome_arquivo)
    amostra.SetCurrentController('MS', 1)
    amostra.SetMassTolerance(userDefined=True, massTolerance=6.0, units=1)
    
    controle = MSFileReader(caminho_controle)
    controle.SetCurrentController('MS', 1)
    controle.SetMassTolerance(userDefined=True, massTolerance=6.0, units=1)

    controleR = MSFileReader(caminho_controle_reinj)
    controleR.SetCurrentController('MS', 1)
    controleR.SetMassTolerance(userDefined=True, massTolerance=6.0, units=1)

    controleN = MSFileReader(caminho_controle_negativo)
    controleN.SetCurrentController('MS', 1)
    controleN.SetMassTolerance(userDefined=True, massTolerance=6.0, units=1)
    
    
    AISPropil =   obter_dados_e_processar(amostra, 5.88, 6.48,'223.1176112863-223.1202887137', MSp)
    AISEfedri =   obter_dados_e_processar(amostra, 3.6, 4.2,'169.14045515118-169.14248484882', MSp)
    AISMefruND3 = obter_dados_e_processar(amostra, 6.4, 7,'384.05163567636-384.05624432364', MSn)
    AISMefruPD3 = obter_dados_e_processar(amostra, 6.4, 7,'386.06617358906-386.07080641094', MSp)
    AISTesto =    obter_dados_e_processar(amostra, 7.97, 8.57,'292.23325658994-292.23676341006', MSp)
    AISTHC =      obter_dados_e_processar(amostra, 9.25, 9.85,'348.22277065084-348.22694934916', MSp)
    AISxAndro =   obter_dados_e_processar(amostra, 8.05, 8.65,'465.24549851026-465.25108148974', MSn)

    CISPropil =   obter_dados_e_processar(controle, 5.88, 6.48,'223.1176112863-223.1202887137', MSp)
    CISEfedri =   obter_dados_e_processar(controle, 3.6, 4.2,'169.14045515118-169.14248484882', MSp)
    CISMefruND3 = obter_dados_e_processar(controle, 6.4, 7,'384.05163567636-384.05624432364', MSn)
    CISMefruPD3 = obter_dados_e_processar(controle, 6.4, 7,'386.06617358906-386.07080641094', MSp)
    CISTesto =    obter_dados_e_processar(controle, 7.97, 8.57,'292.23325658994-292.23676341006', MSp)
    CISTHC =      obter_dados_e_processar(controle, 9.25, 9.85,'348.22277065084-348.22694934916', MSp)
    CISxAndro =   obter_dados_e_processar(controle, 8.05, 8.65,'465.24549851026-465.25108148974', MSn)

    NISPropil =   obter_dados_e_processar(controleN, 5.88, 6.48,'223.1176112863-223.1202887137', MSp)
    NISEfedri =   obter_dados_e_processar(controleN, 3.6, 4.2,'169.14045515118-169.14248484882', MSp)
    NISMefruND3 = obter_dados_e_processar(controleN, 6.4, 7,'384.05163567636-384.05624432364', MSn)
    NISMefruPD3 = obter_dados_e_processar(controleN, 6.4, 7,'386.06617358906-386.07080641094', MSp)
    NISTesto =    obter_dados_e_processar(controleN, 7.97, 8.57,'292.23325658994-292.23676341006', MSp)
    NISTHC =      obter_dados_e_processar(controleN, 9.25, 9.85,'348.22277065084-348.22694934916', MSp)
    NISxAndro =   obter_dados_e_processar(controleN, 8.05, 8.65,'465.24549851026-465.25108148974', MSn)

    RISPropil =   obter_dados_e_processar(controleR, 5.88, 6.48,'223.1176112863-223.1202887137', MSp)
    RISEfedri =   obter_dados_e_processar(controleR, 3.6, 4.2,'169.14045515118-169.14248484882', MSp)
    RISMefruND3 = obter_dados_e_processar(controleR, 6.4, 7,'384.05163567636-384.05624432364', MSn)
    RISMefruPD3 = obter_dados_e_processar(controleR, 6.4, 7,'386.06617358906-386.07080641094', MSp)
    RISTesto =    obter_dados_e_processar(controleR, 7.97, 8.57,'292.23325658994-292.23676341006', MSp)
    RISTHC =      obter_dados_e_processar(controleR, 9.25, 9.85,'348.22277065084-348.22694934916', MSp)
    RISxAndro =   obter_dados_e_processar(controleR, 8.05, 8.65,'465.24549851026-465.25108148974', MSn)
   
    coeficientes_padroes_internos = [
            plotar_e_comparar(AISPropil, CISPropil, NISPropil, RISPropil,'AISPropil', 'CISPropil','NISPropil', 'RISPropil', arquivo_amostra, 847168, 'IS. 7-Propilteofilina_>5.0e6'),
            plotar_e_comparar(AISEfedri, CISEfedri, NISEfedri, RISEfedri, 'AISEfedri', 'CISEfedri', 'NISEfedri', 'RISEfedri', arquivo_amostra, 16213493, 'IS. Efedrina-D3_>1.0e7'),
            plotar_e_comparar(AISMefruND3, CISMefruND3, NISMefruND3, RISMefruND3, 'AISMefruND3', 'CISMefruND3', 'NISMefruND3', 'RISMefruND3', arquivo_amostra, 139024803, 'IS. Mefrusida-D3 (-)_>1.0e6'),
            plotar_e_comparar(AISMefruPD3, CISMefruPD3, NISMefruPD3, RISMefruPD3, 'AISMefruPD3', 'CISMefruPD3', 'NISMefruPD3', 'RISMefruPD3', arquivo_amostra, 139024803, 'IS. Mefrusida-D3 (+)_>1.0e6'),
            plotar_e_comparar(AISTesto, CISTesto, NISTesto, RISTesto, 'AISTesto', 'CISTesto', 'NISTesto', 'RISTesto', arquivo_amostra, 6013, 'IS. Testosterona-D3_>1.0e6'),
            plotar_e_comparar(AISTHC, CISTHC, NISTHC, RISTHC, 'AISTHC', 'CISTHC','NISTHC', 'RISTHC', arquivo_amostra, 76969399, 'IS. THC-COOH-D3'),
            plotar_e_comparar(AISxAndro, CISxAndro, NISxAndro, RISxAndro,'AISxAndro', 'CISxAndro', 'NISxAndro', 'RISxAndro', arquivo_amostra, 443078, 'ISx. Androst-Glic/ Etiocol-Glicuronideo(DS)_>5.0e5')
    ]

    substancias = True
    for coef in coeficientes_padroes_internos:
        if coef is not None:  # Verificar se o coeficiente não é None
            coef_int = float(coef)  # Converter a string em um inteiro
            if coef_int <= -12:
                substancias = False
                break
# region de substâncias
    fluoroanfetamina = obter_dados_e_processar(amostra,4.1,4.9,'154.1017253841-154.1035746159', MSp)
    ACB = obter_dados_e_processar(amostra,1.5,2.3,'283.9554962568-283.9589037432', MSn)
    Acetazolamida = obter_dados_e_processar(amostra,3.2,4.0,'220.9795241149-220.9821758851', MSn)
    Adrafinil = obter_dados_e_processar(amostra,6.35,7.15,'288.06826158006-288.07171841994', MSn)
    Altiazida = obter_dados_e_processar(amostra,5.75,6.55,'381.97392814268-381.97851185732', MSn)
    Amifenazol = obter_dados_e_processar(amostra,3.5,4.3,'192.05783764606-192.06014235394', MSp)
    Amilorida = obter_dados_e_processar(amostra,3.75,4.55,'230.05377966904-230.05654033096', MSp)
    Anastrozol = obter_dados_e_processar(amostra,6.17,6.97,'294.16955497208-294.17308502792', MSp)
    Azosemida = obter_dados_e_processar(amostra,6.75,7.55,'368.99674600624-369.00117399376', MSn)
    Bemetizida = obter_dados_e_processar(amostra,6.45,7.25,'400.0162998878-400.0211001122', MSn)
    BendroflumetiazidaNH4 = obter_dados_e_processar(amostra,6.33,7.13,'439.06897557034-439.07424442966', MSp)
    Bendroflumetiazida = obter_dados_e_processar(amostra,6.33,7.13,'422.04252772964-422.04759227036', MSp)
    Benfluorex = obter_dados_e_processar(amostra,6.65,7.45,'352.14977708866-352.15400291134', MSp)
    Benzfetamina = obter_dados_e_processar(amostra,5.6,6.4,'240.17323895192-240.17612104808', MSp)
    Benzilhidroclorotiazida = obter_dados_e_processar(amostra,5.96,6.76,'386.0018339751-386.0064660249', MSn)
    Benztiazida = obter_dados_e_processar(amostra,6.35,7.15,'429.97364014268-429.97879985732', MSn)
    Betametasona = obter_dados_e_processar(amostra,7.25,8.05,'393.20482075692-393.20953924308', MSp)
    Brinzolamida = obter_dados_e_processar(amostra,4.35,5.15,'384.0692955704-384.0739044296', MSp)
    Bromantano6OH = obter_dados_e_processar(amostra,8.4,9.2,'322.0781675194-322.0820324806', MSp)
    Bromantano = obter_dados_e_processar(amostra,10.3,11.1,'306.08335348886-306.08702651114', MSp)
    Bumetanida = obter_dados_e_processar(amostra,7.35,8.15,'365.1144093004-365.1187906996', MSp)
    ButiazidaP = obter_dados_e_processar(amostra,6.0,6.8,'354.0322257939-354.0364742061', MSp)
    ButiazidaN = obter_dados_e_processar(amostra,6.0,6.8,'352.0176878812-352.0219121188', MSn)
    Ciclesonida = obter_dados_e_processar(amostra,10.22,11.02,'541.31273210412-541.31922789588', MSp)
    Ciclopentiazida = obter_dados_e_processar(amostra,6.7,7.5,'378.0331817873-378.0377182127', MSn)
    Ciclotiazida = obter_dados_e_processar(amostra,6.42,7.22,'388.01746188126-388.02211811874', MSn)
    CisMefentanila = obter_dados_e_processar(amostra,5.95,6.75,'351.24098254146-351.24519745854', MSp)
    Clobenzorex = obter_dados_e_processar(amostra,6.05,6.85,'260.1184892797-260.1216107203', MSp)
    Clobenzorex4OH = obter_dados_e_processar(amostra,5.3,6.1,'276.1132433106-276.1165566894', MSp)
    Clomifeno4OH = obter_dados_e_processar(amostra,7.1,7.9,'422.18559687122-422.19066312878', MSp)
    ClomifenoDESETIL = obter_dados_e_processar(amostra,7.55,8.35,'378.15965102848-378.16418897152', MSp)
    ClomifenoCIS = obter_dados_e_processar(amostra,7.6,8.4,'406.19078284068-406.19565715932', MSp)
    Clopamida = obter_dados_e_processar(amostra,5.95,6.75,'344.08204549534-344.08617450466', MSn)
    Clorazanil = obter_dados_e_processar(amostra,6.3,7.1,'222.0527676754-222.0554323246', MSp)
    Clorotiazida = obter_dados_e_processar(amostra,2.4,3.2,'293.9397863507-293.9433136493', MSn)
    Clortalidona = obter_dados_e_processar(amostra,5.4,6.2,'337.00350796682-337.00755203318', MSn)
    Conivaptan = obter_dados_e_processar(amostra,6.5,7.3,'499.2098547229-499.2158452771', MSp)
    Cropropamida = obter_dados_e_processar(amostra,6.65,7.45,'263.171420962-263.174579038', MSp)
    Crotetamida = obter_dados_e_processar(amostra,6.0,6.8,'249.1558050562-249.1587949438', MSp)
    Desmopressina = obter_dados_e_processar(amostra,5.3,6.1,'535.2175386755-535.2239613245', MSp)
    Dextromoramida = obter_dados_e_processar(amostra,6.55,7.35,'393.2512904781-393.2560095219', MSp)
    Diclorfenamida = obter_dados_e_processar(amostra,4.8,5.6,'302.90551255602-302.90914744398', MSn)
    Dorzolamida = obter_dados_e_processar(amostra,1.5,2.3,'325.03253979306-325.03644020694', MSp)
    Efaproxiral = obter_dados_e_processar(amostra,7.7,8.5,'340.15338906742-340.15747093258', MSn)
    Epitizida = obter_dados_e_processar(amostra,5.75,6.55,'423.9454063123-423.9504936877', MSn)
    Eplerenona6bOH = obter_dados_e_processar(amostra,5.9,6.7,'431.20383276148-431.20900723852', MSp)
    Eplerenona = obter_dados_e_processar(amostra,6.45,7.25,'415.20901873094-415.21400126906', MSp)
    Famprofazona = obter_dados_e_processar(amostra,8.05,8.85,'378.25172047606-378.25625952394', MSp)
    Fembutrazato = obter_dados_e_processar(amostra,7.45,8.25,'368.21981066788-368.22422933212', MSp)
    Fencamina = obter_dados_e_processar(amostra,4.9,5.7,'385.2323385921-385.2369614079', MSp)
    Fenfluramina = obter_dados_e_processar(amostra,5.4,6.2,'204.09823540324-204.10068459676', MSp)
    Fentanil = obter_dados_e_processar(amostra,5.75,6.55,'337.22541663536-337.22946336464', MSp)
    Fludrocortisona = obter_dados_e_processar(amostra,6.85,7.65,'381.20489275692-381.20946724308', MSp)
    Fluocortolona = obter_dados_e_processar(amostra,7.55,8.35,'421.19956278746-421.20461721254', MSn)
    FluticasonaFuroato = obter_dados_e_processar(amostra,8.27,9.07,'539.16773497418-539.17420502582', MSp)
    FluticasonaFuroato17 = obter_dados_e_processar(amostra,7.75,8.55,'491.18464287446-491.19053712554', MSp)
    FluticasonaPropionato = obter_dados_e_processar(amostra,8.25,9.05,'501.1886928498-501.1947071502', MSp)
    Formoterol = obter_dados_e_processar(amostra,4.97,5.77,'345.17880891472-345.18295108528', MSp)
    Fulvestrant = obter_dados_e_processar(amostra,9.37,10.17,'605.30459815062-605.31186184938', MSp)
    Furosemida = obter_dados_e_processar(amostra,6.15,6.95,'328.99791600066-329.00186399934', MSn)
    Gestrinona = obter_dados_e_processar(amostra,7.7,8.5,'309.18305489054-309.18676510946', MSp)
    Heptaminol = obter_dados_e_processar(amostra,2.7,3.5,'146.15306307636-146.15481692364', MSp)
    Hidroclorotiazida = obter_dados_e_processar(amostra,2.8,3.6,'295.9554242568-295.9589757432', MSn)
    Hidroflumetiazida = obter_dados_e_processar(amostra,3.9,4.7,'329.98158009864-329.98553990136', MSn)
    Indacaterol = obter_dados_e_processar(amostra,6.65,7.45,'393.21491069638-393.21962930362', MSp)
    Lisdexanfetamina = obter_dados_e_processar(amostra,3.45,4.25,'264.20545475776-264.20862524224', MSp)
    Lixivaptan = obter_dados_e_processar(amostra,8.35,9.15,'474.13506517254-474.14075482746', MSp)
    Meclofenoxato = obter_dados_e_processar(amostra,6.37,7.27,'184.99888000006-185.00109999994', MSn)
    Mefenorex = obter_dados_e_processar(amostra,5.1,5.9,'212.1187772797-212.1213227203', MSp)
    Mefrusida = obter_dados_e_processar(amostra,6.3,7.1,'381.03282378934-381.03739621066', MSn)
    Mesocarb = obter_dados_e_processar(amostra,6.55,7.35,'339.14313512898-339.14720487102', MSp)
    Metadona = obter_dados_e_processar(amostra,6.65,7.45,'310.21467870076-310.21840129924', MSp)
    MetazolamidaP = obter_dados_e_processar(amostra,4.35,5.15,'237.00963793364-237.01248206636', MSp)
    MetazolamidaN = obter_dados_e_processar(amostra,4.35,5.15,'234.995090021-234.997909979', MSn)
    Meticlotiazida = obter_dados_e_processar(amostra,5.35,6.15,'357.94737230288-357.95166769712', MSn)
    Metiltrienolona = obter_dados_e_processar(amostra,7.65,8.45,'285.18319889054-285.18662110946', MSp)
    Modafinil = obter_dados_e_processar(amostra,6.45,7.25,'296.06979357058-296.07334642942', MSp)
    ModafinilicoFrag = obter_dados_e_processar(amostra,6.6,7.4,'167.084497487-167.086502513', MSn)
    MometasonaFuro = obter_dados_e_processar(amostra,8.25,9.05,'521.14609310468-521.15234689532', MSp)
    Mozavaptan = obter_dados_e_processar(amostra,5.85,6.65,'428.2306806005-428.2358193995', MSp)
    Olodaterol = obter_dados_e_processar(amostra,5.38,6.18,'387.1891268513-387.1937731487', MSp)
    Osilodrostat = obter_dados_e_processar(amostra,3.85,4.65,'228.0917814411-228.0945185589', MSp)
    Pentetrazol = obter_dados_e_processar(amostra,4.05,4.85,'139.09698541308-139.09865458692', MSp)
    Piretanida = obter_dados_e_processar(amostra,7.05,7.85,'361.08420348178-361.08853651822', MSn)
    Politiazida = obter_dados_e_processar(amostra,6.35,7.15,'437.9609722184-437.9662277816', MSn)
    Prenilamina = obter_dados_e_processar(amostra,7.0,7.8,'330.21964867022-330.22361132978', MSp)
    Probenecida = obter_dados_e_processar(amostra,7.45,8.25,'284.0944954228-284.0979045772', MSn)
    Procaterol = obter_dados_e_processar(amostra,3.9,4.7,'291.16857297808-291.17206702192', MSp)
    Prolintano = obter_dados_e_processar(amostra,5.4,6.2,'218.18902085802-218.19163914198', MSp)
    Prostanozol = obter_dados_e_processar(amostra,8.2,9.0,'313.22556063536-313.22931936464', MSp)
    Quinetazona = obter_dados_e_processar(amostra,4.47,5.27,'290.03432978358-290.03781021642', MSp)
    Raloxifeno = obter_dados_e_processar(amostra,6.05,6.85,'474.17051495984-474.17620504016', MSp)
    Ritodrina = obter_dados_e_processar(amostra,4.01,4.81,'288.15769104348-288.16114895652', MSp)
    Salmeterol = obter_dados_e_processar(amostra,7.0,7.8,'416.27704232276-416.28203767724', MSp)
    ASARMACP = obter_dados_e_processar(amostra,8.2,9.0,'291.12411324484-291.12760675516', MSp)
    ASARMAndarina = obter_dados_e_processar(amostra,7.3,8.1,'440.10484935506-440.11013064494', MSn)
    ASARMOstarina = obter_dados_e_processar(amostra,7.52,8.32,'388.0891214513-388.0937785487', MSn)
    ASARMRAD = obter_dados_e_processar(amostra,7.36,8.16,'348.06367160544-348.06784839456', MSn)
    ASARMs1 = obter_dados_e_processar(amostra,8.25,9.05,'401.0741935404-401.0790064596', MSn)
    ASARMs23 = obter_dados_e_processar(amostra,8.3,9.1,'415.04531971314-415.05030028686', MSn)
    ASR9009m2 = obter_dados_e_processar(amostra,4.75,5.55,'314.1150152986-314.1187847014', MSp)
    ASR9009m6 = obter_dados_e_processar(amostra,5.55,6.35,'283.0285518185-283.0319481815', MSp)
    ASR9011 = obter_dados_e_processar(amostra,9.33,10.13,'479.18493487314-479.19068512686', MSp)
    Sufentanil = obter_dados_e_processar(amostra,6.3,7.1,'387.20775673952-387.21240326048', MSp)
    TamoxifenoCarboxi = obter_dados_e_processar(amostra,6.37,7.17,'402.20395676178-402.20878323822', MSp)
    Tamoxifeno = obter_dados_e_processar(amostra,7.92,8.72,'372.22995660686-372.23442339314', MSp)
    Tolvaptan = obter_dados_e_processar(amostra,7.95,8.75,'449.15994502416-449.16533497584', MSp)
    Torasemida = obter_dados_e_processar(amostra,6.25,7.05,'347.11624729002-347.12041270998', MSn)
    Toremifeno = obter_dados_e_processar(amostra,7.05,7.85,'438.1804209017-438.1856790983', MSp)
    Trembolona = obter_dados_e_processar(amostra,7.58,8.38,'271.1676229845-271.1708770155', MSp)
    TriancinolonaAcetonida = obter_dados_e_processar(amostra,6.4,7.2,'451.20995272404-451.21536727596', MSp)
    TriancinolonaFlunisolida = obter_dados_e_processar(amostra,7.38,8.18,'435.21512869356-435.22035130644', MSp)
    Triancinolona = obter_dados_e_processar(amostra,6.15,6.95,'393.16843097526-393.17314902474', MSn)
    Triclormetiazida = obter_dados_e_processar(amostra,5.2,6.0,'379.8896706483-379.8942293517', MSn)
    Vilanterol = obter_dados_e_processar(amostra,6.65,7.45,'486.1779329149-486.1837670851', MSp)
    Xipamida = obter_dados_e_processar(amostra,6.95,7.75,'353.03471177902-353.03894822098', MSn)
#endregion SUBSTANCIAS por controle

#region SUBSTANCIAS por controle
    C4fluo = obter_dados_e_processar(controle, 4.1, 4.9, '154.1017253841-154.1035746159', MSp)
    CACB = obter_dados_e_processar(controle, 1.5, 2.3, '283.9554962568-283.9589037432', MSn)
    CAcetaz = obter_dados_e_processar(controle, 3.2, 4.0, '220.9795241149-220.9821758851', MSn)
    CAcido = obter_dados_e_processar(controle, 7.65, 8.45, '301.00218397606-301.00579602394', MSn)
    CAcidoEtacr = obter_dados_e_processar(controle, 7.65, 8.45, '242.997042009-242.999957991', MSn)
    CAdrafi = obter_dados_e_processar(controle, 6.35, 7.15, '288.06826158006-288.07171841994', MSn)
    CAltiaz = obter_dados_e_processar(controle, 5.75, 6.55, '381.97392814268-381.97851185732', MSn)
    CAmifen = obter_dados_e_processar(controle, 3.5, 4.3, '192.05783764606-192.06014235394', MSp)
    CAmilor = obter_dados_e_processar(controle, 3.75, 4.55, '230.05377966904-230.05654033096', MSp)
    CAnastr = obter_dados_e_processar(controle, 6.17, 6.97, '294.16955497208-294.17308502792', MSp)
    CATFB = obter_dados_e_processar(controle, 3.4, 4.2, '317.98055210524-317.98436789476', MSn)
    CAzosem = obter_dados_e_processar(controle, 6.75, 7.55, '368.99674600624-369.00117399376', MSn)
    CBemeti = obter_dados_e_processar(controle, 6.45, 7.25, '400.0162998878-400.0211001122', MSn)
    CBendro = obter_dados_e_processar(controle, 6.33, 7.13, '439.06897557034-439.07424442966', MSp)
    CBendroflu = obter_dados_e_processar(controle, 6.33, 7.13, '422.04252772964-422.04759227036', MSp)
    CBenflu = obter_dados_e_processar(controle, 6.65, 7.45, '352.14977708866-352.15400291134', MSp)
    CBenzfe = obter_dados_e_processar(controle, 5.6, 6.4, '240.17323895192-240.17612104808', MSp)
    CBenzil = obter_dados_e_processar(controle, 5.96, 6.76, '386.0018339751-386.0064660249', MSn)
    CBenzti = obter_dados_e_processar(controle, 6.35, 7.15, '429.97364014268-429.97879985732', MSn)
    CBetame = obter_dados_e_processar(controle, 7.25, 8.05, '393.20482075692-393.20953924308', MSp)
    CBrinzo = obter_dados_e_processar(controle, 4.35, 5.15, '384.0692955704-384.0739044296', MSp)
    CBroman = obter_dados_e_processar(controle, 8.4, 9.2, '322.0781675194-322.0820324806', MSp)
    CBromanta = obter_dados_e_processar(controle, 10.3, 11.1, '306.08335348886-306.08702651114', MSp)
    CBumeta = obter_dados_e_processar(controle, 7.35, 8.15, '365.1144093004-365.1187906996', MSp)
    CButiaz = obter_dados_e_processar(controle, 6.0, 6.8, '354.0322257939-354.0364742061', MSp)
    CButiazida = obter_dados_e_processar(controle, 6.0, 6.8, '352.0176878812-352.0219121188', MSn)
    CCicles = obter_dados_e_processar(controle, 10.22, 11.02, '541.31273210412-541.31922789588', MSp)
    CCiclop = obter_dados_e_processar(controle, 6.7, 7.5, '378.0331817873-378.0377182127', MSn)
    CCiclot = obter_dados_e_processar(controle, 6.42, 7.22, '388.01746188126-388.02211811874', MSn)
    CCisMe = obter_dados_e_processar(controle, 5.95, 6.75, '351.24098254146-351.24519745854', MSp)
    CClobenzo = obter_dados_e_processar(controle, 6.05, 6.85, '260.1184892797-260.1216107203', MSp)
    CClobenzo4 = obter_dados_e_processar(controle, 5.3, 6.1, '276.1132433106-276.1165566894', MSp)
    CClomif = obter_dados_e_processar(controle, 7.1, 7.9, '422.18559687122-422.19066312878', MSp)
    CClomifeno = obter_dados_e_processar(controle, 7.55, 8.35, '378.15965102848-378.16418897152', MSp)
    CClomifT = obter_dados_e_processar(controle, 7.6, 8.4, '406.19078284068-406.19565715932', MSp)
    CClopam = obter_dados_e_processar(controle, 5.95, 6.75, '344.08204549534-344.08617450466', MSn)
    CCloraz = obter_dados_e_processar(controle, 6.3, 7.1, '222.0527676754-222.0554323246', MSp)
    CClorot = obter_dados_e_processar(controle, 2.4, 3.2, '293.9397863507-293.9433136493', MSn)
    CClorta = obter_dados_e_processar(controle, 5.4, 6.2, '337.00350796682-337.00755203318', MSn)
    CConiva = obter_dados_e_processar(controle, 6.5, 7.3, '499.2098547229-499.2158452771', MSp)
    CCropro = obter_dados_e_processar(controle, 6.65, 7.45, '263.171420962-263.174579038', MSp)
    CCrotet = obter_dados_e_processar(controle, 6.0, 6.8, '249.1558050562-249.1587949438', MSp)
    CDesmop = obter_dados_e_processar(controle, 5.3, 6.1, '535.2175386755-535.2239613245', MSp)
    CDextro = obter_dados_e_processar(controle, 6.55, 7.35, '393.2512904781-393.2560095219', MSp)
    CDiclor = obter_dados_e_processar(controle, 4.8, 5.6, '302.90551255602-302.90914744398', MSn)
    CDorzol = obter_dados_e_processar(controle, 1.5, 2.3, '325.03253979306-325.03644020694', MSp)
    CEfapro = obter_dados_e_processar(controle, 7.7, 8.5, '340.15338906742-340.15747093258', MSn)
    CEpitiz = obter_dados_e_processar(controle, 5.75, 6.55, '423.9454063123-423.9504936877', MSn)
    CEplere = obter_dados_e_processar(controle, 5.9, 6.7, '431.20383276148-431.20900723852', MSp)
    CEplerenona = obter_dados_e_processar(controle, 6.45, 7.25, '415.20901873094-415.21400126906', MSp)
    CFampro = obter_dados_e_processar(controle, 8.05, 8.85, '378.25172047606-378.25625952394', MSp)
    CFembut = obter_dados_e_processar(controle, 7.45, 8.25, '368.21981066788-368.22422933212', MSp)
    CFencam = obter_dados_e_processar(controle, 4.9, 5.7, '385.2323385921-385.2369614079', MSp)
    CFenflu = obter_dados_e_processar(controle, 5.4, 6.2, '204.09823540324-204.10068459676', MSp)
    CFentan = obter_dados_e_processar(controle, 5.75, 6.55, '337.22541663536-337.22946336464', MSp)
    CFludro = obter_dados_e_processar(controle, 6.85, 7.65, '381.20489275692-381.20946724308', MSp)
    CFluocortol = obter_dados_e_processar(controle, 7.55, 8.35, '421.19956278746-421.20461721254', MSn)
    CFlutic = obter_dados_e_processar(controle, 8.27, 9.07, '539.16773497418-539.17420502582', MSp)
    CFluticFuro = obter_dados_e_processar(controle, 7.75, 8.55, '491.18464287446-491.19053712554', MSp)
    CFluticPro = obter_dados_e_processar(controle, 8.25, 9.05, '501.1886928498-501.1947071502', MSp)
    CFormo = obter_dados_e_processar(controle, 4.97, 5.77, '345.17880891472-345.18295108528', MSp)
    CFulves = obter_dados_e_processar(controle, 9.37, 10.17, '605.30459815062-605.31186184938', MSp)
    CFurose = obter_dados_e_processar(controle, 6.15, 6.95, '328.99791600066-329.00186399934', MSn)
    CGestri = obter_dados_e_processar(controle, 7.7, 8.5, '309.18305489054-309.18676510946', MSp)
    CHeptam = obter_dados_e_processar(controle, 2.7, 3.5, '146.15306307636-146.15481692364', MSp)
    CHidroc = obter_dados_e_processar(controle, 2.8, 3.6, '295.9554242568-295.9589757432', MSn)
    CHidrof = obter_dados_e_processar(controle, 3.9, 4.7, '329.98158009864-329.98553990136', MSn)
    CIndaca = obter_dados_e_processar(controle, 6.65, 7.45, '393.21491069638-393.21962930362', MSp)
    CLisdex = obter_dados_e_processar(controle, 3.45, 4.25, '264.20545475776-264.20862524224', MSp)
    CLixiva = obter_dados_e_processar(controle, 8.35, 9.15, '474.13506517254-474.14075482746', MSp)
    CMeclof = obter_dados_e_processar(controle, 6.37, 7.27, '184.99888000006-185.00109999994', MSn)
    CMefeno = obter_dados_e_processar(controle, 5.1, 5.9, '212.1187772797-212.1213227203', MSp)
    CMefrusida = obter_dados_e_processar(controle, 6.3, 7.1, '381.03282378934-381.03739621066', MSn)
    CMesoca = obter_dados_e_processar(controle, 6.55, 7.35, '339.14313512898-339.14720487102', MSp)
    CMetado = obter_dados_e_processar(controle, 6.65, 7.45, '310.21467870076-310.21840129924', MSp)
    CMetazo = obter_dados_e_processar(controle, 4.35, 5.15, '237.00963793364-237.01248206636', MSp)
    CMetazola = obter_dados_e_processar(controle, 4.35, 5.15, '234.995090021-234.997909979', MSn)
    CMeticl = obter_dados_e_processar(controle, 5.35, 6.15, '357.94737230288-357.95166769712', MSn)
    CMetilt = obter_dados_e_processar(controle, 7.65, 8.45, '285.18319889054-285.18662110946', MSp)
    CModafi = obter_dados_e_processar(controle, 6.45, 7.25, '296.06979357058-296.07334642942', MSp)
    CModafiFrag = obter_dados_e_processar(controle, 6.6, 7.4, '167.084497487-167.086502513', MSn)
    CMometaFuro = obter_dados_e_processar(controle, 8.25, 9.05, '521.14609310468-521.15234689532', MSp)
    CMozava = obter_dados_e_processar(controle, 5.85, 6.65, '428.2306806005-428.2358193995', MSp)
    COlodat = obter_dados_e_processar(controle, 5.38, 6.18, '387.1891268513-387.1937731487', MSp)
    COsilod = obter_dados_e_processar(controle, 3.85, 4.65, '228.0917814411-228.0945185589', MSp)
    CPentet = obter_dados_e_processar(controle, 4.05, 4.85, '139.09698541308-139.09865458692', MSp)
    CPireta = obter_dados_e_processar(controle, 7.05, 7.85, '361.08420348178-361.08853651822', MSn)
    CPoliti = obter_dados_e_processar(controle, 6.35, 7.15, '437.9609722184-437.9662277816', MSn)
    CPrenil = obter_dados_e_processar(controle, 7.0, 7.8, '330.21964867022-330.22361132978', MSp)
    CProben = obter_dados_e_processar(controle, 7.45, 8.25, '284.0944954228-284.0979045772', MSn)
    CProcat = obter_dados_e_processar(controle, 3.9, 4.7, '291.16857297808-291.17206702192', MSp)
    CProlin = obter_dados_e_processar(controle, 5.4, 6.2, '218.18902085802-218.19163914198', MSp)
    CProsta = obter_dados_e_processar(controle, 8.2, 9.0, '313.22556063536-313.22931936464', MSp)
    CQuinet = obter_dados_e_processar(controle, 4.47, 5.27, '290.03432978358-290.03781021642', MSp)
    CRaloxi = obter_dados_e_processar(controle, 6.05, 6.85, '474.17051495984-474.17620504016', MSp)
    CRitodr = obter_dados_e_processar(controle, 4.01, 4.81, '288.15769104348-288.16114895652', MSp)
    CSalmet = obter_dados_e_processar(controle, 7.0, 7.8, '416.27704232276-416.28203767724', MSp)
    CSARMACP = obter_dados_e_processar(controle, 8.2, 9.0, '291.12411324484-291.12760675516', MSp)
    CSARMAndarina = obter_dados_e_processar(controle, 7.3, 8.1, '440.10484935506-440.11013064494', MSn)
    CSARMOstarina = obter_dados_e_processar(controle, 7.52, 8.32, '388.0891214513-388.0937785487', MSn)
    CSARMRAD = obter_dados_e_processar(controle, 7.36, 8.16, '348.06367160544-348.06784839456', MSn)
    CSARMs1 = obter_dados_e_processar(controle, 8.25, 9.05, '401.0741935404-401.0790064596', MSn)
    CSARMs23 = obter_dados_e_processar(controle, 8.3, 9.1, '415.04531971314-415.05030028686', MSn)
    CSR9009m2 = obter_dados_e_processar(controle, 4.75, 5.55, '314.1150152986-314.1187847014', MSp)
    CSR9009m6 = obter_dados_e_processar(controle, 5.55, 6.35, '283.0285518185-283.0319481815', MSp)
    CSR9011 = obter_dados_e_processar(controle, 9.33, 10.13, '479.18493487314-479.19068512686', MSp)
    CSufent = obter_dados_e_processar(controle, 6.3, 7.1, '387.20775673952-387.21240326048', MSp)
    CTamoxi = obter_dados_e_processar(controle, 6.37, 7.17, '402.20395676178-402.20878323822', MSp)
    CTamoxifeno = obter_dados_e_processar(controle, 7.92, 8.72, '372.22995660686-372.23442339314', MSp)
    CTolvap = obter_dados_e_processar(controle, 7.95, 8.75, '449.15994502416-449.16533497584', MSp)
    CTorase = obter_dados_e_processar(controle, 6.25, 7.05, '347.11624729002-347.12041270998', MSn)
    CToremi = obter_dados_e_processar(controle, 7.05, 7.85, '438.1804209017-438.1856790983', MSp)
    CTrembo = obter_dados_e_processar(controle, 7.58, 8.38, '271.1676229845-271.1708770155', MSp)
    CTriancAce = obter_dados_e_processar(controle, 6.4, 7.2, '451.20995272404-451.21536727596', MSp)
    CTriancFlu = obter_dados_e_processar(controle, 7.38, 8.18, '435.21512869356-435.22035130644', MSp)
    CTriancino = obter_dados_e_processar(controle, 6.15, 6.95, '393.16843097526-393.17314902474', MSn)
    CTriclo = obter_dados_e_processar(controle, 5.2, 6.0, '379.8896706483-379.8942293517', MSn)
    CVilant = obter_dados_e_processar(controle, 6.65, 7.45, '486.1779329149-486.1837670851', MSp)
    CXipami = obter_dados_e_processar(controle, 6.95, 7.75, '353.03471177902-353.03894822098', MSn)


#endregion
    
#region SUBSTANCIAS por controleNegativo
    N4fluo = obter_dados_e_processar(controleN, 4.1, 4.9, '154.1017253841-154.1035746159', MSp)
    NACB = obter_dados_e_processar(controleN, 1.5, 2.3, '283.9554962568-283.9589037432', MSn)
    NAcetaz = obter_dados_e_processar(controleN, 3.2, 4.0, '220.9795241149-220.9821758851', MSn)
    NAcido = obter_dados_e_processar(controleN, 7.65, 8.45, '301.00218397606-301.00579602394', MSn)
    NAcidoEtacr = obter_dados_e_processar(controleN, 7.65, 8.45, '242.997042009-242.999957991', MSn)
    NAdrafi = obter_dados_e_processar(controleN, 6.35, 7.15, '288.06826158006-288.07171841994', MSn)
    NAltiaz = obter_dados_e_processar(controleN, 5.75, 6.55, '381.97392814268-381.97851185732', MSn)
    NAmifen = obter_dados_e_processar(controleN, 3.5, 4.3, '192.05783764606-192.06014235394', MSp)
    NAmilor = obter_dados_e_processar(controleN, 3.75, 4.55, '230.05377966904-230.05654033096', MSp)
    NAnastr = obter_dados_e_processar(controleN, 6.17, 6.97, '294.16955497208-294.17308502792', MSp)
    NATFB = obter_dados_e_processar(controleN, 3.4, 4.2, '317.98055210524-317.98436789476', MSn)
    NAzosem = obter_dados_e_processar(controleN, 6.75, 7.55, '368.99674600624-369.00117399376', MSn)
    NBemeti = obter_dados_e_processar(controleN, 6.45, 7.25, '400.0162998878-400.0211001122', MSn)
    NBendro = obter_dados_e_processar(controleN, 6.33, 7.13, '439.06897557034-439.07424442966', MSp)
    NBendroflu = obter_dados_e_processar(controleN, 6.33, 7.13, '422.04252772964-422.04759227036', MSp)
    NBenflu = obter_dados_e_processar(controleN, 6.65, 7.45, '352.14977708866-352.15400291134', MSp)
    NBenzfe = obter_dados_e_processar(controleN, 5.6, 6.4, '240.17323895192-240.17612104808', MSp)
    NBenzil = obter_dados_e_processar(controleN, 5.96, 6.76, '386.0018339751-386.0064660249', MSn)
    NBenzti = obter_dados_e_processar(controleN, 6.35, 7.15, '429.97364014268-429.97879985732', MSn)
    NBetame = obter_dados_e_processar(controleN, 7.25, 8.05, '393.20482075692-393.20953924308', MSp)
    NBrinzo = obter_dados_e_processar(controleN, 4.35, 5.15, '384.0692955704-384.0739044296', MSp)
    NBroman = obter_dados_e_processar(controleN, 8.4, 9.2, '322.0781675194-322.0820324806', MSp)
    NBromanta = obter_dados_e_processar(controleN, 10.3, 11.1, '306.08335348886-306.08702651114', MSp)
    NBumeta = obter_dados_e_processar(controleN, 7.35, 8.15, '365.1144093004-365.1187906996', MSp)
    NButiaz = obter_dados_e_processar(controleN, 6.0, 6.8, '354.0322257939-354.0364742061', MSp)
    NButiazida = obter_dados_e_processar(controleN, 6.0, 6.8, '352.0176878812-352.0219121188', MSn)
    NCicles = obter_dados_e_processar(controleN, 10.22, 11.02, '541.31273210412-541.31922789588', MSp)
    NCiclop = obter_dados_e_processar(controleN, 6.7, 7.5, '378.0331817873-378.0377182127', MSn)
    NCiclot = obter_dados_e_processar(controleN, 6.42, 7.22, '388.01746188126-388.02211811874', MSn)
    NCisMe = obter_dados_e_processar(controleN, 5.95, 6.75, '351.24098254146-351.24519745854', MSp)
    NClobenzo = obter_dados_e_processar(controleN, 6.05, 6.85, '260.1184892797-260.1216107203', MSp)
    NClobenzo4 = obter_dados_e_processar(controleN, 5.3, 6.1, '276.1132433106-276.1165566894', MSp)
    NClomif = obter_dados_e_processar(controleN, 7.1, 7.9, '422.18559687122-422.19066312878', MSp)
    NClomifeno = obter_dados_e_processar(controleN, 7.55, 8.35, '378.15965102848-378.16418897152', MSp)
    NClomifT = obter_dados_e_processar(controleN, 7.6, 8.4, '406.19078284068-406.19565715932', MSp)
    NClopam = obter_dados_e_processar(controleN, 5.95, 6.75, '344.08204549534-344.08617450466', MSn)
    NCloraz = obter_dados_e_processar(controleN, 6.3, 7.1, '222.0527676754-222.0554323246', MSp)
    NClorot = obter_dados_e_processar(controleN, 2.4, 3.2, '293.9397863507-293.9433136493', MSn)
    NClorta = obter_dados_e_processar(controleN, 5.4, 6.2, '337.00350796682-337.00755203318', MSn)
    NConiva = obter_dados_e_processar(controleN, 6.5, 7.3, '499.2098547229-499.2158452771', MSp)
    NCropro = obter_dados_e_processar(controleN, 6.65, 7.45, '263.171420962-263.174579038', MSp)
    NCrotet = obter_dados_e_processar(controleN, 6.0, 6.8, '249.1558050562-249.1587949438', MSp)
    NDesmop = obter_dados_e_processar(controleN, 5.3, 6.1, '535.2175386755-535.2239613245', MSp)
    NDextro = obter_dados_e_processar(controleN, 6.55, 7.35, '393.2512904781-393.2560095219', MSp)
    NDiclor = obter_dados_e_processar(controleN, 4.8, 5.6, '302.90551255602-302.90914744398', MSn)
    NDorzol = obter_dados_e_processar(controleN, 1.5, 2.3, '325.03253979306-325.03644020694', MSp)
    NEfapro = obter_dados_e_processar(controleN, 7.7, 8.5, '340.15338906742-340.15747093258', MSn)
    NEpitiz = obter_dados_e_processar(controleN, 5.75, 6.55, '423.9454063123-423.9504936877', MSn)
    NEplere = obter_dados_e_processar(controleN, 5.9, 6.7, '431.20383276148-431.20900723852', MSp)
    NEplerenona = obter_dados_e_processar(controleN, 6.45, 7.25, '415.20901873094-415.21400126906', MSp)
    NFampro = obter_dados_e_processar(controleN, 8.05, 8.85, '378.25172047606-378.25625952394', MSp)
    NFembut = obter_dados_e_processar(controleN, 7.45, 8.25, '368.21981066788-368.22422933212', MSp)
    NFencam = obter_dados_e_processar(controleN, 4.9, 5.7, '385.2323385921-385.2369614079', MSp)
    NFenflu = obter_dados_e_processar(controleN, 5.4, 6.2, '204.09823540324-204.10068459676', MSp)
    NFentan = obter_dados_e_processar(controleN, 5.75, 6.55, '337.22541663536-337.22946336464', MSp)
    NFludro = obter_dados_e_processar(controleN, 6.85, 7.65, '381.20489275692-381.20946724308', MSp)
    NFluocortol = obter_dados_e_processar(controleN, 7.55, 8.35, '421.19956278746-421.20461721254', MSn)
    NFlutic = obter_dados_e_processar(controleN, 8.27, 9.07, '539.16773497418-539.17420502582', MSp)
    NFluticFuro = obter_dados_e_processar(controleN, 7.75, 8.55, '491.18464287446-491.19053712554', MSp)
    NFluticPro = obter_dados_e_processar(controleN, 8.25, 9.05, '501.1886928498-501.1947071502', MSp)
    NFormo = obter_dados_e_processar(controleN, 4.97, 5.77, '345.17880891472-345.18295108528', MSp)
    NFulves = obter_dados_e_processar(controleN, 9.37, 10.17, '605.30459815062-605.31186184938', MSp)
    NFurose = obter_dados_e_processar(controleN, 6.15, 6.95, '328.99791600066-329.00186399934', MSn)
    NGestri = obter_dados_e_processar(controleN, 7.7, 8.5, '309.18305489054-309.18676510946', MSp)
    NHeptam = obter_dados_e_processar(controleN, 2.7, 3.5, '146.15306307636-146.15481692364', MSp)
    NHidroc = obter_dados_e_processar(controleN, 2.8, 3.6, '295.9554242568-295.9589757432', MSn)
    NHidrof = obter_dados_e_processar(controleN, 3.9, 4.7, '329.98158009864-329.98553990136', MSn)
    NIndaca = obter_dados_e_processar(controleN, 6.65, 7.45, '393.21491069638-393.21962930362', MSp)
    NLisdex = obter_dados_e_processar(controleN, 3.45, 4.25, '264.20545475776-264.20862524224', MSp)
    NLixiva = obter_dados_e_processar(controleN, 8.35, 9.15, '474.13506517254-474.14075482746', MSp)
    NMeclof = obter_dados_e_processar(controleN, 6.37, 7.27, '184.99888000006-185.00109999994', MSn)
    NMefeno = obter_dados_e_processar(controleN, 5.1, 5.9, '212.1187772797-212.1213227203', MSp)
    NMefrusida = obter_dados_e_processar(controleN, 6.3, 7.1, '381.03282378934-381.03739621066', MSn)
    NMesoca = obter_dados_e_processar(controleN, 6.55, 7.35, '339.14313512898-339.14720487102', MSp)
    NMetado = obter_dados_e_processar(controleN, 6.65, 7.45, '310.21467870076-310.21840129924', MSp)
    NMetazo = obter_dados_e_processar(controleN, 4.35, 5.15, '237.00963793364-237.01248206636', MSp)
    NMetazola = obter_dados_e_processar(controleN, 4.35, 5.15, '234.995090021-234.997909979', MSn)
    NMeticl = obter_dados_e_processar(controleN, 5.35, 6.15, '357.94737230288-357.95166769712', MSn)
    NMetilt = obter_dados_e_processar(controleN, 7.65, 8.45, '285.18319889054-285.18662110946', MSp)
    NModafi = obter_dados_e_processar(controleN, 6.45, 7.25, '296.06979357058-296.07334642942', MSp)
    NModafiFrag = obter_dados_e_processar(controleN, 6.6, 7.4, '167.084497487-167.086502513', MSn)
    NMometaFuro = obter_dados_e_processar(controleN, 8.25, 9.05, '521.14609310468-521.15234689532', MSp)
    NMozava = obter_dados_e_processar(controleN, 5.85, 6.65, '428.2306806005-428.2358193995', MSp)
    NOlodat = obter_dados_e_processar(controleN, 5.38, 6.18, '387.1891268513-387.1937731487', MSp)
    NOsilod = obter_dados_e_processar(controleN, 3.85, 4.65, '228.0917814411-228.0945185589', MSp)
    NPentet = obter_dados_e_processar(controleN, 4.05, 4.85, '139.09698541308-139.09865458692', MSp)
    NPireta = obter_dados_e_processar(controleN, 7.05, 7.85, '361.08420348178-361.08853651822', MSn)
    NPoliti = obter_dados_e_processar(controleN, 6.35, 7.15, '437.9609722184-437.9662277816', MSn)
    NPrenil = obter_dados_e_processar(controleN, 7.0, 7.8, '330.21964867022-330.22361132978', MSp)
    NProben = obter_dados_e_processar(controleN, 7.45, 8.25, '284.0944954228-284.0979045772', MSn)
    NProcat = obter_dados_e_processar(controleN, 3.9, 4.7, '291.16857297808-291.17206702192', MSp)
    NProlin = obter_dados_e_processar(controleN, 5.4, 6.2, '218.18902085802-218.19163914198', MSp)
    NProsta = obter_dados_e_processar(controleN, 8.2, 9.0, '313.22556063536-313.22931936464', MSp)
    NQuinet = obter_dados_e_processar(controleN, 4.47, 5.27, '290.03432978358-290.03781021642', MSp)
    NRaloxi = obter_dados_e_processar(controleN, 6.05, 6.85, '474.17051495984-474.17620504016', MSp)
    NRitodr = obter_dados_e_processar(controleN, 4.01, 4.81, '288.15769104348-288.16114895652', MSp)
    NSalmet = obter_dados_e_processar(controleN, 7.0, 7.8, '416.27704232276-416.28203767724', MSp)
    NSARMACP = obter_dados_e_processar(controleN, 8.2, 9.0, '291.12411324484-291.12760675516', MSp)
    NSARMAndarina = obter_dados_e_processar(controleN, 7.3, 8.1, '440.10484935506-440.11013064494', MSn)
    NSARMOstarina = obter_dados_e_processar(controleN, 7.52, 8.32, '388.0891214513-388.0937785487', MSn)
    NSARMRAD = obter_dados_e_processar(controleN, 7.36, 8.16, '348.06367160544-348.06784839456', MSn)
    NSARMs1 = obter_dados_e_processar(controleN, 8.25, 9.05, '401.0741935404-401.0790064596', MSn)
    NSARMs23 = obter_dados_e_processar(controleN, 8.3, 9.1, '415.04531971314-415.05030028686', MSn)
    NSR9009m2 = obter_dados_e_processar(controleN, 4.75, 5.55, '314.1150152986-314.1187847014', MSp)
    NSR9009m6 = obter_dados_e_processar(controleN, 5.55, 6.35, '283.0285518185-283.0319481815', MSp)
    NSR9011 = obter_dados_e_processar(controleN, 9.33, 10.13, '479.18493487314-479.19068512686', MSp)
    NSufent = obter_dados_e_processar(controleN, 6.3, 7.1, '387.20775673952-387.21240326048', MSp)
    NTamoxi = obter_dados_e_processar(controleN, 6.37, 7.17, '402.20395676178-402.20878323822', MSp)
    NTamoxifeno = obter_dados_e_processar(controleN, 7.92, 8.72, '372.22995660686-372.23442339314', MSp)
    NTolvap = obter_dados_e_processar(controleN, 7.95, 8.75, '449.15994502416-449.16533497584', MSp)
    NTorase = obter_dados_e_processar(controleN, 6.25, 7.05, '347.11624729002-347.12041270998', MSn)
    NToremi = obter_dados_e_processar(controleN, 7.05, 7.85, '438.1804209017-438.1856790983', MSp)
    NTrembo = obter_dados_e_processar(controleN, 7.58, 8.38, '271.1676229845-271.1708770155', MSp)
    NTriancAce = obter_dados_e_processar(controleN, 6.4, 7.2, '451.20995272404-451.21536727596', MSp)
    NTriancFlu = obter_dados_e_processar(controleN, 7.38, 8.18, '435.21512869356-435.22035130644', MSp)
    NTriancino = obter_dados_e_processar(controleN, 6.15, 6.95, '393.16843097526-393.17314902474', MSn)
    NTriclo = obter_dados_e_processar(controleN, 5.2, 6.0, '379.8896706483-379.8942293517', MSn)
    NVilant = obter_dados_e_processar(controleN, 6.65, 7.45, '486.1779329149-486.1837670851', MSp)
    NXipami = obter_dados_e_processar(controleN, 6.95, 7.75, '353.03471177902-353.03894822098', MSn)

#endregion   

#region SUBSTANCIAS por controleReinj
    R4fluo = obter_dados_e_processar(controleR, 4.1, 4.9, '154.1017253841-154.1035746159', MSp)
    RACB = obter_dados_e_processar(controleR, 1.5, 2.3, '283.9554962568-283.9589037432', MSn)
    RAcetaz = obter_dados_e_processar(controleR, 3.2, 4.0, '220.9795241149-220.9821758851', MSn)
    RAcido = obter_dados_e_processar(controleR, 7.65, 8.45, '301.00218397606-301.00579602394', MSn)
    RAcidoEtacr = obter_dados_e_processar(controleR, 7.65, 8.45, '242.997042009-242.999957991', MSn)
    RAdrafi = obter_dados_e_processar(controleR, 6.35, 7.15, '288.06826158006-288.07171841994', MSn)
    RAltiaz = obter_dados_e_processar(controleR, 5.75, 6.55, '381.97392814268-381.97851185732', MSn)
    RAmifen = obter_dados_e_processar(controleR, 3.5, 4.3, '192.05783764606-192.06014235394', MSp)
    RAmilor = obter_dados_e_processar(controleR, 3.75, 4.55, '230.05377966904-230.05654033096', MSp)
    RAnastr = obter_dados_e_processar(controleR, 6.17, 6.97, '294.16955497208-294.17308502792', MSp)
    RATFB = obter_dados_e_processar(controleR, 3.4, 4.2, '317.98055210524-317.98436789476', MSn)
    RAzosem = obter_dados_e_processar(controleR, 6.75, 7.55, '368.99674600624-369.00117399376', MSn)
    RBemeti = obter_dados_e_processar(controleR, 6.45, 7.25, '400.0162998878-400.0211001122', MSn)
    RBendro = obter_dados_e_processar(controleR, 6.33, 7.13, '439.06897557034-439.07424442966', MSp)
    RBendroflu = obter_dados_e_processar(controleR, 6.33, 7.13, '422.04252772964-422.04759227036', MSp)
    RBenflu = obter_dados_e_processar(controleR, 6.65, 7.45, '352.14977708866-352.15400291134', MSp)
    RBenzfe = obter_dados_e_processar(controleR, 5.6, 6.4, '240.17323895192-240.17612104808', MSp)
    RBenzil = obter_dados_e_processar(controleR, 5.96, 6.76, '386.0018339751-386.0064660249', MSn)
    RBenzti = obter_dados_e_processar(controleR, 6.35, 7.15, '429.97364014268-429.97879985732', MSn)
    RBetame = obter_dados_e_processar(controleR, 7.25, 8.05, '393.20482075692-393.20953924308', MSp)
    RBrinzo = obter_dados_e_processar(controleR, 4.35, 5.15, '384.0692955704-384.0739044296', MSp)
    RBroman = obter_dados_e_processar(controleR, 8.4, 9.2, '322.0781675194-322.0820324806', MSp)
    RBromanta = obter_dados_e_processar(controleR, 10.3, 11.1, '306.08335348886-306.08702651114', MSp)
    RBumeta = obter_dados_e_processar(controleR, 7.35, 8.15, '365.1144093004-365.1187906996', MSp)
    RButiaz = obter_dados_e_processar(controleR, 6.0, 6.8, '354.0322257939-354.0364742061', MSp)
    RButiazida = obter_dados_e_processar(controleR, 6.0, 6.8, '352.0176878812-352.0219121188', MSn)
    RCicles = obter_dados_e_processar(controleR, 10.22, 11.02, '541.31273210412-541.31922789588', MSp)
    RCiclop = obter_dados_e_processar(controleR, 6.7, 7.5, '378.0331817873-378.0377182127', MSn)
    RCiclot = obter_dados_e_processar(controleR, 6.42, 7.22, '388.01746188126-388.02211811874', MSn)
    RCisMe = obter_dados_e_processar(controleR, 5.95, 6.75, '351.24098254146-351.24519745854', MSp)
    RClobenzo = obter_dados_e_processar(controleR, 6.05, 6.85, '260.1184892797-260.1216107203', MSp)
    RClobenzo4 = obter_dados_e_processar(controleR, 5.3, 6.1, '276.1132433106-276.1165566894', MSp)
    RClomif = obter_dados_e_processar(controleR, 7.1, 7.9, '422.18559687122-422.19066312878', MSp)
    RClomifeno = obter_dados_e_processar(controleR, 7.55, 8.35, '378.15965102848-378.16418897152', MSp)
    RClomifT = obter_dados_e_processar(controleR, 7.6, 8.4, '406.19078284068-406.19565715932', MSp)
    RClopam = obter_dados_e_processar(controleR, 5.95, 6.75, '344.08204549534-344.08617450466', MSn)
    RCloraz = obter_dados_e_processar(controleR, 6.3, 7.1, '222.0527676754-222.0554323246', MSp)
    RClorot = obter_dados_e_processar(controleR, 2.4, 3.2, '293.9397863507-293.9433136493', MSn)
    RClorta = obter_dados_e_processar(controleR, 5.4, 6.2, '337.00350796682-337.00755203318', MSn)
    RConiva = obter_dados_e_processar(controleR, 6.5, 7.3, '499.2098547229-499.2158452771', MSp)
    RCropro = obter_dados_e_processar(controleR, 6.65, 7.45, '263.171420962-263.174579038', MSp)
    RCrotet = obter_dados_e_processar(controleR, 6.0, 6.8, '249.1558050562-249.1587949438', MSp)
    RDesmop = obter_dados_e_processar(controleR, 5.3, 6.1, '535.2175386755-535.2239613245', MSp)
    RDextro = obter_dados_e_processar(controleR, 6.55, 7.35, '393.2512904781-393.2560095219', MSp)
    RDiclor = obter_dados_e_processar(controleR, 4.8, 5.6, '302.90551255602-302.90914744398', MSn)
    RDorzol = obter_dados_e_processar(controleR, 1.5, 2.3, '325.03253979306-325.03644020694', MSp)
    REfapro = obter_dados_e_processar(controleR, 7.7, 8.5, '340.15338906742-340.15747093258', MSn)
    REpitiz = obter_dados_e_processar(controleR, 5.75, 6.55, '423.9454063123-423.9504936877', MSn)
    REplere = obter_dados_e_processar(controleR, 5.9, 6.7, '431.20383276148-431.20900723852', MSp)
    REplerenona = obter_dados_e_processar(controleR, 6.45, 7.25, '415.20901873094-415.21400126906', MSp)
    RFampro = obter_dados_e_processar(controleR, 8.05, 8.85, '378.25172047606-378.25625952394', MSp)
    RFembut = obter_dados_e_processar(controleR, 7.45, 8.25, '368.21981066788-368.22422933212', MSp)
    RFencam = obter_dados_e_processar(controleR, 4.9, 5.7, '385.2323385921-385.2369614079', MSp)
    RFenflu = obter_dados_e_processar(controleR, 5.4, 6.2, '204.09823540324-204.10068459676', MSp)
    RFentan = obter_dados_e_processar(controleR, 5.75, 6.55, '337.22541663536-337.22946336464', MSp)
    RFludro = obter_dados_e_processar(controleR, 6.85, 7.65, '381.20489275692-381.20946724308', MSp)
    RFluocortol = obter_dados_e_processar(controleR, 7.55, 8.35, '421.19956278746-421.20461721254', MSn)
    RFlutic = obter_dados_e_processar(controleR, 8.27, 9.07, '539.16773497418-539.17420502582', MSp)
    RFluticFuro = obter_dados_e_processar(controleR, 7.75, 8.55, '491.18464287446-491.19053712554', MSp)
    RFluticPro = obter_dados_e_processar(controleR, 8.25, 9.05, '501.1886928498-501.1947071502', MSp)
    RFormo = obter_dados_e_processar(controleR, 4.97, 5.77, '345.17880891472-345.18295108528', MSp)
    RFulves = obter_dados_e_processar(controleR, 9.37, 10.17, '605.30459815062-605.31186184938', MSp)
    RFurose = obter_dados_e_processar(controleR, 6.15, 6.95, '328.99791600066-329.00186399934', MSn)
    RGestri = obter_dados_e_processar(controleR, 7.7, 8.5, '309.18305489054-309.18676510946', MSp)
    RHeptam = obter_dados_e_processar(controleR, 2.7, 3.5, '146.15306307636-146.15481692364', MSp)
    RHidroc = obter_dados_e_processar(controleR, 2.8, 3.6, '295.9554242568-295.9589757432', MSn)
    RHidrof = obter_dados_e_processar(controleR, 3.9, 4.7, '329.98158009864-329.98553990136', MSn)
    RIndaca = obter_dados_e_processar(controleR, 6.65, 7.45, '393.21491069638-393.21962930362', MSp)
    RLisdex = obter_dados_e_processar(controleR, 3.45, 4.25, '264.20545475776-264.20862524224', MSp)
    RLixiva = obter_dados_e_processar(controleR, 8.35, 9.15, '474.13506517254-474.14075482746', MSp)
    RMeclof = obter_dados_e_processar(controleR, 6.42, 7.22, '184.99888000006-185.00109999994', MSn)
    RMefeno = obter_dados_e_processar(controleR, 5.1, 5.9, '212.1187772797-212.1213227203', MSp)
    RMefrusida = obter_dados_e_processar(controleR, 6.3, 7.1, '381.03282378934-381.03739621066', MSn)
    RMesoca = obter_dados_e_processar(controleR, 6.55, 7.35, '339.14313512898-339.14720487102', MSp)
    RMetado = obter_dados_e_processar(controleR, 6.65, 7.45, '310.21467870076-310.21840129924', MSp)
    RMetazo = obter_dados_e_processar(controleR, 4.35, 5.15, '237.00963793364-237.01248206636', MSp)
    RMetazola = obter_dados_e_processar(controleR, 4.35, 5.15, '234.995090021-234.997909979', MSn)
    RMeticl = obter_dados_e_processar(controleR, 5.35, 6.15, '357.94737230288-357.95166769712', MSn)
    RMetilt = obter_dados_e_processar(controleR, 7.65, 8.45, '285.18319889054-285.18662110946', MSp)
    RModafi = obter_dados_e_processar(controleR, 6.45, 7.25, '296.06979357058-296.07334642942', MSp)
    RModafiFrag = obter_dados_e_processar(controleR, 6.6, 7.4, '167.084497487-167.086502513', MSn)
    RMometaFuro = obter_dados_e_processar(controleR, 8.25, 9.05, '521.14609310468-521.15234689532', MSp)
    RMozava = obter_dados_e_processar(controleR, 5.85, 6.65, '428.2306806005-428.2358193995', MSp)
    ROlodat = obter_dados_e_processar(controleR, 5.38, 6.18, '387.1891268513-387.1937731487', MSp)
    ROsilod = obter_dados_e_processar(controleR, 3.85, 4.65, '228.0917814411-228.0945185589', MSp)
    RPentet = obter_dados_e_processar(controleR, 4.05, 4.85, '139.09698541308-139.09865458692', MSp)
    RPireta = obter_dados_e_processar(controleR, 7.05, 7.85, '361.08420348178-361.08853651822', MSn)
    RPoliti = obter_dados_e_processar(controleR, 6.35, 7.15, '437.9609722184-437.9662277816', MSn)
    RPrenil = obter_dados_e_processar(controleR, 7.0, 7.8, '330.21964867022-330.22361132978', MSp)
    RProben = obter_dados_e_processar(controleR, 7.45, 8.25, '284.0944954228-284.0979045772', MSn)
    RProcat = obter_dados_e_processar(controleR, 3.9, 4.7, '291.16857297808-291.17206702192', MSp)
    RProlin = obter_dados_e_processar(controleR, 5.4, 6.2, '218.18902085802-218.19163914198', MSp)
    RProsta = obter_dados_e_processar(controleR, 8.2, 9.0, '313.22556063536-313.22931936464', MSp)
    RQuinet = obter_dados_e_processar(controleR, 4.47, 5.27, '290.03432978358-290.03781021642', MSp)
    RRaloxi = obter_dados_e_processar(controleR, 6.05, 6.85, '474.17051495984-474.17620504016', MSp)
    RRitodr = obter_dados_e_processar(controleR, 4.01, 4.81, '288.15769104348-288.16114895652', MSp)
    RSalmet = obter_dados_e_processar(controleR, 7.0, 7.8, '416.27704232276-416.28203767724', MSp)
    RSARMACP = obter_dados_e_processar(controleR, 8.2, 9.0, '291.12411324484-291.12760675516', MSp)
    RSARMAndarina = obter_dados_e_processar(controleR, 7.3, 8.1, '440.10484935506-440.11013064494', MSn)
    RSARMOstarina = obter_dados_e_processar(controleR, 7.52, 8.32, '388.0891214513-388.0937785487', MSn)
    RSARMRAD = obter_dados_e_processar(controleR, 7.36, 8.16, '348.06367160544-348.06784839456', MSn)
    RSARMs1 = obter_dados_e_processar(controleR, 8.25, 9.05, '401.0741935404-401.0790064596', MSn)
    RSARMs23 = obter_dados_e_processar(controleR, 8.3, 9.1, '415.04531971314-415.05030028686', MSn)
    RSR9009m2 = obter_dados_e_processar(controleR, 4.75, 5.55, '314.1150152986-314.1187847014', MSp)
    RSR9009m6 = obter_dados_e_processar(controleR, 5.55, 6.35, '283.0285518185-283.0319481815', MSp)
    RSR9011 = obter_dados_e_processar(controleR, 9.33, 10.13, '479.18493487314-479.19068512686', MSp)
    RSufent = obter_dados_e_processar(controleR, 6.3, 7.1, '387.20775673952-387.21240326048', MSp)
    RTamoxi = obter_dados_e_processar(controleR, 6.37, 7.17, '402.20395676178-402.20878323822', MSp)
    RTamoxifeno = obter_dados_e_processar(controleR, 7.92, 8.72, '372.22995660686-372.23442339314', MSp)
    RTolvap = obter_dados_e_processar(controleR, 7.95, 8.75, '449.15994502416-449.16533497584', MSp)
    RTorase = obter_dados_e_processar(controleR, 6.25, 7.05, '347.11624729002-347.12041270998', MSn)
    RToremi = obter_dados_e_processar(controleR, 7.05, 7.85, '438.1804209017-438.1856790983', MSp)
    RTrembo = obter_dados_e_processar(controleR, 7.58, 8.38, '271.1676229845-271.1708770155', MSp)
    RTriancAce = obter_dados_e_processar(controleR, 6.4, 7.2, '451.20995272404-451.21536727596', MSp)
    RTriancFlu = obter_dados_e_processar(controleR, 7.38, 8.18, '435.21512869356-435.22035130644', MSp)
    RTriancino = obter_dados_e_processar(controleR, 6.15, 6.95, '393.16843097526-393.17314902474', MSn)
    RTriclo = obter_dados_e_processar(controleR, 5.2, 6.0, '379.8896706483-379.8942293517', MSn)
    RVilant = obter_dados_e_processar(controleR, 6.65, 7.45, '486.1779329149-486.1837670851', MSp)
    RXipami = obter_dados_e_processar(controleR, 6.95, 7.75, '353.03471177902-353.03894822098', MSn)
#endregion  
    
#region Adiciona as plotagens ao laudo
    if substancias:

        plotar_e_comparar(fluoroanfetamina, C4fluo, N4fluo, R4fluo, 'fluoroanfetamina', 'C4fluo', 'N4fluo', 'R4fluo', arquivo_amostra, 9986, 'S6. 4-fluoroanfetamina*')
        plotar_e_comparar(ACB, CACB, NACB, RACB, 'ACB', 'CACB', 'NACB', 'RACB', arquivo_amostra, 67136, 'S5. ACB (4-amino-6-cloro-1,3-benzenodisulfonamida)*')
        plotar_e_comparar(Acetazolamida, CAcetaz, NAcetaz, RAcetaz, 'Acetazolamida', 'CAcetaz', 'NAcetaz', 'RAcetaz', arquivo_amostra, 1986, 'S5. Acetazolamida*')
        plotar_e_comparar(Adrafinil, CAdrafi, NAdrafi, RAdrafi, 'Adrafinil', 'CAdrafi', 'NAdrafi', 'RAdrafi', arquivo_amostra, 3033226, 'S6. Adrafinil*')
        plotar_e_comparar(Altiazida, CAltiaz, NAltiaz, RAltiaz, 'Altiazida', 'CAltiaz', 'NAltiaz', 'RAltiaz', arquivo_amostra, 2122, 'S5. Altiazida*')
        plotar_e_comparar(Amifenazol, CAmifen, NAmifen, RAmifen, 'Amifenazol', 'CAmifen', 'NAmifen', 'RAmifen', arquivo_amostra, 10275, 'S6. Amifenazol*')
        plotar_e_comparar(Amilorida, CAmilor, NAmilor, RAmilor, 'Amilorida', 'CAmilor', 'NAmilor', 'RAmilor', arquivo_amostra, 16231, 'S5. Amilorida*')
        plotar_e_comparar(Anastrozol, CAnastr, NAnastr, RAnastr, 'Anastrozol', 'CAnastr', 'NAnastr', 'RAnastr', arquivo_amostra, 2187, 'S4. Anastrozol*')
        plotar_e_comparar(Azosemida, CAzosem, NAzosem, RAzosem, 'Azosemida', 'CAzosem', 'NAzosem', 'RAzosem', arquivo_amostra, 2273, 'S5. Azosemida*')
        plotar_e_comparar(Bemetizida, CBemeti, NBemeti, RBemeti, 'Bemetizida', 'CBemeti', 'NBemeti', 'RBemeti', arquivo_amostra, 72070, 'S5. Bemetizida*')
        plotar_e_comparar(BendroflumetiazidaNH4, CBendro, NBendro, RBendro, 'BendroflumetiazidaNH4', 'CBendro', 'NBendro', 'RBendro', arquivo_amostra, 69561, 'S5. Bendroflumetiazida-NH4*')
        plotar_e_comparar(Bendroflumetiazida, CBendroflu, NBendroflu, RBendroflu, 'Bendroflumetiazida', 'CBendroflu', 'NBendroflu', 'RBendroflu', arquivo_amostra, 2315, 'S5. Bendroflumetiazida*')
        plotar_e_comparar(Benfluorex, CBenflu, NBenflu, RBenflu, 'Benfluorex', 'CBenflu', 'NBenflu', 'RBenflu', arquivo_amostra, 2318, 'S6. Benfluorex*')
        plotar_e_comparar(Benzfetamina, CBenzfe, NBenzfe, RBenzfe, 'Benzfetamina', 'CBenzfe', 'NBenzfe', 'RBenzfe', arquivo_amostra, 5311017, 'S6. Benzfetamina*')
        plotar_e_comparar(Benzilhidroclorotiazida, CBenzil, NBenzil, RBenzil, 'Benzilhidroclorotiazida', 'CBenzil', 'NBenzil', 'RBenzil', arquivo_amostra, 2348, 'S5. Benzilhidroclorotiazida*')
        plotar_e_comparar(Benztiazida, CBenzti, NBenzti, RBenzti, 'Benztiazida', 'CBenzti', 'NBenzti', 'RBenzti', arquivo_amostra, 343, 'S5. Benztiazida*')
        plotar_e_comparar(Betametasona, CBetame, NBetame, RBetame, 'Betametasona', 'CBetame', 'NBetame', 'RBetame', arquivo_amostra, 9782, 'S9. Betametasona*')
        plotar_e_comparar(Brinzolamida, CBrinzo, NBrinzo, RBrinzo, 'Brinzolamida', 'CBrinzo', 'NBrinzo', 'RBrinzo', arquivo_amostra, 68844, 'S5. Brinzolamida*')
        plotar_e_comparar(Bromantano6OH, CBroman, NBroman, RBroman, 'Bromantano6OH', 'CBroman', 'NBroman', 'RBroman', arquivo_amostra, 4660557, 'S6. Bromantano (6-OH)*')
        plotar_e_comparar(Bromantano, CBromanta, NBromanta, RBromanta, 'Bromantano', 'CBromanta', 'NBromanta', 'RBromanta', arquivo_amostra, 4660557, 'S6. Bromantano*')
        plotar_e_comparar(Bumetanida, CBumeta, NBumeta, RBumeta, 'Bumetanida', 'CBumeta', 'NBumeta', 'RBumeta', arquivo_amostra, 2471, 'S5. Bumetanida*')
        plotar_e_comparar(ButiazidaP, CButiaz, NButiaz, RButiaz, 'ButiazidaP', 'CButiaz', 'NButiaz', 'RButiaz', arquivo_amostra, 16274, 'S5. Butiazida (+)*')
        plotar_e_comparar(ButiazidaN, CButiazida, NButiazida, RButiazida, 'ButiazidaN', 'CButiazida', 'NButiazida', 'RButiazida', arquivo_amostra, 16274, 'S5. Butiazida (-)*')
        plotar_e_comparar(Ciclesonida, CCicles, NCicles, RCicles, 'Ciclesonida', 'CCicles', 'NCicles', 'RCicles', arquivo_amostra, 6918155, 'S9. Ciclesonida*')
        plotar_e_comparar(Ciclopentiazida, CCiclop, NCiclop, RCiclop, 'Ciclopentiazida', 'CCiclop', 'NCiclop', 'RCiclop', arquivo_amostra, 2904, 'S5. Ciclopentiazida*')
        plotar_e_comparar(Ciclotiazida, CCiclot, NCiclot, RCiclot, 'Ciclotiazida', 'CCiclot', 'NCiclot', 'RCiclot', arquivo_amostra, 2910, 'S5. Ciclotiazida*')
        plotar_e_comparar(CisMefentanila, CCisMe, NCisMe, RCisMe, 'CisMefentanila', 'CCisMe', 'NCisMe', 'RCisMe', arquivo_amostra, 61996, 'S7. Cis-Mefentanila {ou 3-metilfentanil}*')
        plotar_e_comparar(Clobenzorex, CClobenzo, NClobenzo, RClobenzo, 'Clobenzorex', 'CClobenzo', 'NClobenzo', 'RClobenzo', arquivo_amostra, 71675, 'S6. Clobenzorex*')
        plotar_e_comparar(Clobenzorex4OH, CClobenzo4, NClobenzo4, RClobenzo4, 'Clobenzorex4OH', 'CClobenzo4', 'NClobenzo4', 'RClobenzo4', arquivo_amostra, 71675, 'S6. Clobenzorex (4-OH)*')
        plotar_e_comparar(Clomifeno4OH, CClomif, NClomif, RClomif, 'Clomifeno4OH', 'CClomif', 'NClomif', 'RClomif', arquivo_amostra, 2800, 'S4. Clomifeno (4-OH)*/ Toremifeno (4-OH*)')
        plotar_e_comparar(ClomifenoDESETIL, CClomifeno, NClomifeno, RClomifeno, 'ClomifenoDESETIL', 'CClomifeno', 'NClomifeno', 'RClomifeno', arquivo_amostra, 2800, 'S4. Clomifeno (desetil)*')
        plotar_e_comparar(ClomifenoCIS, CClomifT, NClomifT, RClomifT, 'ClomifenoCIS', 'CClomifT', 'NClomifT', 'RClomifT', arquivo_amostra, 2800, 'S4. Clomifeno (cis/trans)*/ Toremifeno')
        plotar_e_comparar(Clopamida, CClopam, NClopam, RClopam, 'Clopamida', 'CClopam', 'NClopam', 'RClopam', arquivo_amostra, 12492, 'S5. Clopamida*')
        plotar_e_comparar(Clorazanil, CCloraz, NCloraz, RCloraz, 'Clorazanil', 'CCloraz', 'NCloraz', 'RCloraz', arquivo_amostra, 10374, 'S5. Clorazanil*')
        plotar_e_comparar(Clorotiazida, CClorot, NClorot, RClorot, 'Clorotiazida', 'CClorot', 'NClorot', 'RClorot', arquivo_amostra, 2720, 'S5. Clorotiazida*')
        plotar_e_comparar(Clortalidona, CClorta, NClorta, RClorta, 'Clortalidona', 'CClorta', 'NClorta', 'RClorta', arquivo_amostra, 2732, 'S5. Clortalidona*')
        plotar_e_comparar(Conivaptan, CConiva, NConiva, RConiva, 'Conivaptan', 'CConiva', 'NConiva', 'RConiva', arquivo_amostra, 151171, 'S5. Conivaptan*')
        plotar_e_comparar(Cropropamida, CCropro, NCropro, RCropro, 'Cropropamida', 'CCropro', 'NCropro', 'RCropro', arquivo_amostra, 5369258, 'S6. Cropropamida-Na*')
        plotar_e_comparar(Crotetamida, CCrotet, NCrotet, RCrotet, 'Crotetamida', 'CCrotet', 'NCrotet', 'RCrotet', arquivo_amostra, 5368010, 'S6. Crotetamida-Na*')
        plotar_e_comparar(Desmopressina, CDesmop, NDesmop, RDesmop, 'Desmopressina', 'CDesmop', 'NDesmop', 'RDesmop', arquivo_amostra, 5311065, 'S5. Desmopressina*')
        plotar_e_comparar(Dextromoramida, CDextro, NDextro, RDextro, 'Dextromoramida', 'CDextro', 'NDextro', 'RDextro', arquivo_amostra, 92943, 'S7. Dextromoramida*')
        plotar_e_comparar(Diclorfenamida, CDiclor, NDiclor, RDiclor, 'Diclorfenamida', 'CDiclor', 'NDiclor', 'RDiclor', arquivo_amostra, 3038, 'S5. Diclorfenamida*')
        plotar_e_comparar(Dorzolamida, CDorzol, NDorzol, RDorzol, 'Dorzolamida', 'CDorzol', 'NDorzol', 'RDorzol', arquivo_amostra, 5284549, 'S5. Dorzolamida*')
        plotar_e_comparar(Epitizida, CEpitiz, NEpitiz, REpitiz, 'Epitizida', 'CEpitiz', 'NEpitiz', 'REpitiz', arquivo_amostra, 15671, 'S5. Epitizida*')
        plotar_e_comparar(Eplerenona6bOH, CEplere, NEplere, REplere, 'Eplerenona6bOH', 'CEplere', 'NEplere', 'REplere', arquivo_amostra, 443872, 'S5. Eplerenona (6b-OH)*')
        plotar_e_comparar(Eplerenona, CEplerenona, NEplerenona, REplerenona, 'Eplerenona', 'CEplerenona', 'NEplerenona', 'REplerenona', arquivo_amostra, 443872, 'S5. Eplerenona*')
        plotar_e_comparar(Famprofazona, CFampro, NFampro, RFampro, 'Famprofazona', 'CFampro', 'NFampro', 'RFampro', arquivo_amostra, 3326, 'S6. Famprofazona*')
        plotar_e_comparar(Fembutrazato, CFembut, NFembut, RFembut, 'Fembutrazato', 'CFembut', 'NFembut', 'RFembut', arquivo_amostra, 20395, 'S6. Fembutrazato*')
        plotar_e_comparar(Fencamina, CFencam, NFencam, RFencam, 'Fencamina', 'CFencam', 'NFencam', 'RFencam', arquivo_amostra, 115374, 'S6. Fencamina*')
        plotar_e_comparar(Fenfluramina, CFenflu, NFenflu, RFenflu, 'Fenfluramina', 'CFenflu', 'NFenflu', 'RFenflu', arquivo_amostra, 3337, 'S6. Fenfluramina (nor)*')
        plotar_e_comparar(Fentanil, CFentan, NFentan, RFentan, 'Fentanil', 'CFentan', 'NFentan', 'RFentan', arquivo_amostra, 3345, 'S7. Fentanil*')
        plotar_e_comparar(Fludrocortisona, CFludro, NFludro, RFludro, 'Fludrocortisona', 'CFludro', 'NFludro', 'RFludro', arquivo_amostra, 31378, 'S9. Fludrocortisona*')
        plotar_e_comparar(Fluocortolona, CFluocortol, NFluocortol, RFluocortol, 'Fluocortolona', 'CFluocortol', 'NFluocortol', 'RFluocortol', arquivo_amostra, 9053, 'S9. Fluocortolona [COOH]*')
        plotar_e_comparar(FluticasonaFuroato, CFlutic, NFlutic, RFlutic, 'FluticasonaFuroato', 'CFlutic', 'NFlutic', 'RFlutic', arquivo_amostra, 5311101, 'S9. Fluticasona furoato*')
        plotar_e_comparar(FluticasonaFuroato17, CFluticFuro, NFluticFuro, RFluticFuro, 'FluticasonaFuroato17', 'CFluticFuro', 'NFluticFuro', 'RFluticFuro', arquivo_amostra, 5311101, 'S9. Fluticasona furoato (17b-ac carboxilico)*')
        plotar_e_comparar(FluticasonaPropionato, CFluticPro, NFluticPro, RFluticPro, 'FluticasonaPropionato', 'CFluticPro', 'NFluticPro', 'RFluticPro', arquivo_amostra, 5311101, 'S9. Fluticasona propionato*')
        plotar_e_comparar(Formoterol, CFormo, NFormo, RFormo, 'Formoterol', 'CFormo', 'NFormo', 'RFormo', arquivo_amostra, 3410, 'S3q. Formoterol*')
        plotar_e_comparar(Fulvestrant, CFulves, NFulves, RFulves, 'Fulvestrant', 'CFulves', 'NFulves', 'RFulves', arquivo_amostra, 104741, 'S4. Fulvestrant (17-ceto)*')
        plotar_e_comparar(Furosemida, CFurose, NFurose, RFurose, 'Furosemida', 'CFurose', 'NFurose', 'RFurose', arquivo_amostra, 3440, 'S5. Furosemida*')
        plotar_e_comparar(Gestrinona, CGestri, NGestri, RGestri, 'Gestrinona', 'CGestri', 'NGestri', 'RGestri', arquivo_amostra, 27812, 'S1. Gestrinona*')
        plotar_e_comparar(Hidroclorotiazida, CHidroc, NHidroc, RHidroc, 'Hidroclorotiazida', 'CHidroc', 'NHidroc', 'RHidroc', arquivo_amostra, 3639, 'S5. Hidroclorotiazida*')
        plotar_e_comparar(Hidroflumetiazida, CHidrof, NHidrof, RHidrof, 'Hidroflumetiazida', 'CHidrof', 'NHidrof', 'RHidrof', arquivo_amostra, 3647, 'S5. Hidroflumetiazida*')
        plotar_e_comparar(Indacaterol, CIndaca, NIndaca, RIndaca, 'Indacaterol', 'CIndaca', 'NIndaca', 'RIndaca', arquivo_amostra, 6918554, 'S3. Indacaterol*')
        plotar_e_comparar(Lisdexanfetamina, CLisdex, NLisdex, RLisdex, 'Lisdexanfetamina', 'CLisdex', 'NLisdex', 'RLisdex', arquivo_amostra, 11597698, 'S6. Lisdexanfetamina*')
        plotar_e_comparar(Lixivaptan, CLixiva, NLixiva, RLixiva, 'Lixivaptan', 'CLixiva', 'NLixiva', 'RLixiva', arquivo_amostra, 172997, 'S5. Lixivaptan*')
        plotar_e_comparar(Meclofenoxato, CMeclof, NMeclof, RMeclof, 'Meclofenoxato', 'CMeclof', 'NMeclof', 'RMeclof', arquivo_amostra, 4039, 'S6. Meclofenoxato M (4-CPA)*')
        plotar_e_comparar(Mefenorex, CMefeno, NMefeno, RMefeno, 'Mefenorex', 'CMefeno', 'NMefeno', 'RMefeno', arquivo_amostra, 21777, 'S6. Mefenorex*')
        plotar_e_comparar(Mefrusida, CMefrusida, NMefrusida, RMefrusida, 'Mefrusida', 'CMefrusida', 'NMefrusida', 'RMefrusida', arquivo_amostra, 4047, 'S5. Mefrusida*')
        plotar_e_comparar(Mesocarb, CMesoca, NMesoca, RMesoca, 'Mesocarb', 'CMesoca', 'NMesoca', 'RMesoca', arquivo_amostra, 9551611, 'S6. Mesocarb (p-OH)*')
        plotar_e_comparar(Metadona, CMetado, NMetado, RMetado, 'Metadona', 'CMetado', 'NMetado', 'RMetado', arquivo_amostra, 4095, 'S7. Metadona*')
        plotar_e_comparar(MetazolamidaP, CMetazo, NMetazo, RMetazo, 'MetazolamidaP', 'CMetazo', 'NMetazo', 'RMetazo', arquivo_amostra, 4100, 'S5. Metazolamida (+)*')
        plotar_e_comparar(MetazolamidaN, CMetazola, NMetazola, RMetazola, 'MetazolamidaN', 'CMetazola', 'NMetazola', 'RMetazola', arquivo_amostra, 4100, 'S5. Metazolamida (-)*')
        plotar_e_comparar(Meticlotiazida, CMeticl, NMeticl, RMeticl, 'Meticlotiazida', 'CMeticl', 'NMeticl', 'RMeticl', arquivo_amostra, 4121, 'S5. Meticlotiazida*')
        plotar_e_comparar(Metiltrienolona, CMetilt, NMetilt, RMetilt, 'Metiltrienolona', 'CMetilt', 'NMetilt', 'RMetilt', arquivo_amostra, 261000, 'S1. Metiltrienolona {ou Metribolona}*')
        plotar_e_comparar(Modafinil, CModafi, NModafi, RModafi, 'Modafinil', 'CModafi', 'NModafi', 'RModafi', arquivo_amostra, 4236, 'S6. Modafinil-Na*')
        plotar_e_comparar(ModafinilicoFrag, CModafiFrag, NModafiFrag, RModafiFrag, 'ModafinilicoFrag', 'CModafiFrag', 'NModafiFrag', 'RModafiFrag', arquivo_amostra, 4236, 'S6. Modafinilico Ac Fragment* Full>167')
        plotar_e_comparar(MometasonaFuro, CMometaFuro, NMometaFuro, RMometaFuro, 'MometasonaFuro', 'CMometaFuro', 'NMometaFuro', 'RMometaFuro', arquivo_amostra, 441335, 'S9. Mometasona Furoato*')
        plotar_e_comparar(Mozavaptan, CMozava, NMozava, RMozava, 'Mozavaptan', 'CMozava', 'NMozava', 'RMozava', arquivo_amostra, 119369, 'S5. Mozavaptan*')
        plotar_e_comparar(Olodaterol, COlodat, NOlodat, ROlodat, 'Olodaterol', 'COlodat', 'NOlodat', 'ROlodat', arquivo_amostra, 11504295, 'S3. Olodaterol*')
        plotar_e_comparar(Osilodrostat, COsilod, NOsilod, ROsilod, 'Osilodrostat', 'COsilod', 'NOsilod', 'ROsilod', arquivo_amostra, 44139752, 'S1. Osilodrostat*')
        plotar_e_comparar(Pentetrazol, CPentet, NPentet, RPentet, 'Pentetrazol', 'CPentet', 'NPentet', 'RPentet', arquivo_amostra, 5917, 'S6. Pentetrazol*')
        plotar_e_comparar(Piretanida, CPireta, NPireta, RPireta, 'Piretanida', 'CPireta', 'NPireta', 'RPireta', arquivo_amostra, 4849, 'S5. Piretanida*')
        plotar_e_comparar(Politiazida, CPoliti, NPoliti, RPoliti, 'Politiazida', 'CPoliti', 'NPoliti', 'RPoliti', arquivo_amostra, 4870, 'S5. Politiazida*')
        plotar_e_comparar(Prenilamina, CPrenil, NPrenil, RPrenil, 'Prenilamina', 'CPrenil', 'NPrenil', 'RPrenil', arquivo_amostra, 9801, 'S6. Prenilamina*')
        plotar_e_comparar(Probenecida, CProben, NProben, RProben, 'Probenecida', 'CProben', 'NProben', 'RProben', arquivo_amostra, 4911, 'S5. Probenecida*')
        plotar_e_comparar(Procaterol, CProcat, NProcat, RProcat, 'Procaterol', 'CProcat', 'NProcat', 'RProcat', arquivo_amostra, 4916, 'S3. Procaterol*')
        plotar_e_comparar(Prolintano, CProlin, NProlin, RProlin, 'Prolintano', 'CProlin', 'NProlin', 'RProlin', arquivo_amostra, 14592, 'S6. Prolintano*')
        plotar_e_comparar(Prostanozol, CProsta, NProsta, RProsta, 'Prostanozol', 'CProsta', 'NProsta', 'RProsta', arquivo_amostra, 56842253, 'S1. Prostanozol (ceto)*')
        plotar_e_comparar(Quinetazona, CQuinet, NQuinet, RQuinet, 'Quinetazona', 'CQuinet', 'NQuinet', 'RQuinet', arquivo_amostra, 6307, 'S5. Quinetazona*')
        plotar_e_comparar(Raloxifeno, CRaloxi, NRaloxi, RRaloxi, 'Raloxifeno', 'CRaloxi', 'NRaloxi', 'RRaloxi', arquivo_amostra, 5035, 'S4. Raloxifeno*')
        plotar_e_comparar(Ritodrina, CRitodr, NRitodr, RRitodr, 'Ritodrina', 'CRitodr', 'NRitodr', 'RRitodr', arquivo_amostra, 688570, 'S3. Ritodrina*')
        plotar_e_comparar(Salmeterol, CSalmet, NSalmet, RSalmet, 'Salmeterol', 'CSalmet', 'NSalmet', 'RSalmet', arquivo_amostra, 5152, 'S3. Salmeterol*')
        plotar_e_comparar(ASARMACP, CSARMACP, NSARMACP, RSARMACP, 'ASARMACP', 'CSARMACP', 'NSARMACP', 'RSARMACP', arquivo_amostra, 11638442, 'S1. SARM: ACP-105*')
        plotar_e_comparar(ASARMAndarina, CSARMAndarina, NSARMAndarina, RSARMAndarina, 'ASARMAndarina', 'CSARMAndarina', 'NSARMAndarina', 'RSARMAndarina', arquivo_amostra, 9824562, 'S1. SARM S4 (Andarina)*')
        plotar_e_comparar(ASARMOstarina, CSARMOstarina, NSARMOstarina, RSARMOstarina, 'ASARMOstarina', 'CSARMOstarina', 'NSARMOstarina', 'RSARMOstarina', arquivo_amostra, 11326715, 'S1. SARM Enobosarm (Ostarina)*')
        plotar_e_comparar(ASARMRAD, CSARMRAD, NSARMRAD, RSARMRAD, 'ASARMRAD', 'CSARMRAD', 'NSARMRAD', 'RSARMRAD', arquivo_amostra, 44200882, 'S1. SARM RAD-140 Fragment* Full>348')
        plotar_e_comparar(ASARMs1, CSARMs1, NSARMs1, RSARMs1, 'ASARMs1', 'CSARMs1', 'NSARMs1', 'RSARMs1', arquivo_amostra, None, 'S1. SARM S1*')
        plotar_e_comparar(ASARMs23, CSARMs23, NSARMs23, RSARMs23, 'ASARMs23', 'CSARMs23', 'NSARMs23', 'RSARMs23', arquivo_amostra, 24892822, 'S1. SARM S-23 (SB19042)*')
        plotar_e_comparar(ASR9009m2, CSR9009m2, NSR9009m2, RSR9009m2, 'ASR9009m2', 'CSR9009m2', 'NSR9009m2', 'RSR9009m2', arquivo_amostra, 57394020, 'S4. SR9009 M2*')
        plotar_e_comparar(ASR9009m6, CSR9009m6, NSR9009m6, RSR9009m6, 'ASR9009m6', 'CSR9009m6', 'NSR9009m6', 'RSR9009m6', arquivo_amostra, None, 'S4. SR9009 M6*')
        plotar_e_comparar(ASR9011, CSR9011, NSR9011, RSR9011, 'ASR9011', 'CSR9011', 'NSR9011', 'RSR9011', arquivo_amostra, 57394021, 'S4. SR9011*')
        plotar_e_comparar(Sufentanil, CSufent, NSufent, RSufent, 'Sufentanil', 'CSufent', 'NSufent', 'RSufent', arquivo_amostra, 41693, 'S7. Sufentanil*')
        plotar_e_comparar(TamoxifenoCarboxi, CTamoxi, NTamoxi, RTamoxi, 'TamoxifenoCarboxi', 'CTamoxi', 'NTamoxi', 'RTamoxi', arquivo_amostra, None, 'S4. Tamoxifeno (carboxi) {ou Toremifeno (carboxi)}*')
        plotar_e_comparar(Tamoxifeno, CTamoxifeno, NTamoxifeno, RTamoxifeno, 'Tamoxifeno', 'CTamoxifeno', 'NTamoxifeno', 'RTamoxifeno', arquivo_amostra, 2733526, 'S4. Tamoxifeno*')
        plotar_e_comparar(Tolvaptan, CTolvap, NTolvap, RTolvap, 'Tolvaptan', 'CTolvap', 'NTolvap', 'RTolvap', arquivo_amostra, 216237, 'S5. Tolvaptan*')
        plotar_e_comparar(Torasemida, CTorase, NTorase, RTorase, 'Torasemida', 'CTorase', 'NTorase', 'RTorase', arquivo_amostra, 41781, 'S5. Torasemida*')
        plotar_e_comparar(Toremifeno, CToremi, NToremi, RToremi, 'Toremifeno', 'CToremi', 'NToremi', 'RToremi', arquivo_amostra, 3005573, 'S4. Toremifeno (OH-metoxi-N-desmetil)*')
        plotar_e_comparar(Trembolona, CTrembo, NTrembo, RTrembo, 'Trembolona', 'CTrembo', 'NTrembo', 'RTrembo', arquivo_amostra, 25015, 'S1. Trembolona/ Epitrembolona*')
        plotar_e_comparar(TriancinolonaAcetonida, CTriancAce, NTriancAce, RTriancAce, 'TriancinolonaAcetonida', 'CTriancAce', 'NTriancAce', 'RTriancAce', arquivo_amostra, 6436, 'S9. Triancinolona acetonida (6b-OH)*')
        plotar_e_comparar(TriancinolonaFlunisolida, CTriancFlu, NTriancFlu, RTriancFlu, 'TriancinolonaFlunisolida', 'CTriancFlu', 'NTriancFlu', 'RTriancFlu', arquivo_amostra, 82153, 'S9. Triancinolona acetonida*/ Flunisolida* Conc 15 e 30')
        plotar_e_comparar(Triancinolona, CTriancino, NTriancino, RTriancino, 'Triancinolona', 'CTriancino', 'NTriancino', 'RTriancino', arquivo_amostra, 31307, 'S9. Triancinolona*')
        plotar_e_comparar(Triclormetiazida, CTriclo, NTriclo, RTriclo, 'Triclormetiazida', 'CTriclo', 'NTriclo', 'RTriclo', arquivo_amostra, 5560, 'S5. Triclormetiazida*')
        plotar_e_comparar(Vilanterol, CVilant, NVilant, RVilant, 'Vilanterol', 'CVilant', 'NVilant', 'RVilant', arquivo_amostra, 10184665, 'S3. Vilanterol*')
        plotar_e_comparar(Xipamida, CXipami, NXipami, RXipami, 'Xipamida', 'CXipami', 'NXipami', 'RXipami', arquivo_amostra, 26618, 'S5. Xipamida*')    


    else:
        print("O TEMPO DE RETENÇÃO DO PADRÃO INTERNO ESTA DESLOCADO")
#endregion  
 

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print("Uso: python dopinho.py <arquivo_amostra> <arquivo_controle> <arquivo_controle_reinj> <arquivo_controle_negativo>")
        sys.exit(1)

    arquivo_amostra = sys.argv[1]
    arquivo_controle = sys.argv[2]
    arquivo_controle_reinj = sys.argv[3]
    arquivo_controle_negativo = sys.argv[4]

    processar_arquivo_amostra(arquivo_amostra, arquivo_controle, arquivo_controle_reinj, arquivo_controle_negativo)

end_time = timeit.default_timer()  
execution_time = end_time - start_time
print(f"O processamento completo levou {execution_time:.2f} segundos.")

libraries = [
    "pubchempy",
    "pymsfilereader",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "numpy",
    "scikit-fuzzy",
    "rdkit",
]

# Imprime o nome e a versão de cada biblioteca
for lib in libraries:
    try:
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib}: Não está instalado")
