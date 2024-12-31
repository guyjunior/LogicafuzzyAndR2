from prov.model import ProvDocument
from prov.dot import prov_to_dot
from IPython.display import Image

# Criação do modelo PROV
doc = ProvDocument()

# Definir um namespace para os identificadores
doc.add_namespace('ex', 'http://example.org/')

# Entidades
dataset_raw = doc.entity('ex:dataset_raw', {'prov:label': 'Dados RAW de HPLC-MS'})
aligned_chromatograms = doc.entity('ex:aligned_chromatograms', {'prov:label': 'Cromatogramas Alinhados'})
pubchem_data = doc.entity('ex:pubchem_data', {'prov:label': 'Dados do PubChem'})
final_report = doc.entity('ex:final_report', {'prov:label': 'Relatório Final de Resultados'})

# Agentes
pipeline_system = doc.agent('ex:pipeline_system', {'prov:label': 'Pipeline Computacional'})
analyst = doc.agent('ex:analyst', {'prov:label': 'Analista do LBCD'})

# Atividades
align_chromatograms = doc.activity('ex:align_chromatograms', startTime='2024-01-01T09:00:00Z', endTime='2024-01-01T09:30:00Z')
fetch_pubchem_data = doc.activity('ex:fetch_pubchem_data', startTime='2024-01-01T09:30:00Z', endTime='2024-01-01T09:40:00Z')
generate_final_report = doc.activity('ex:generate_final_report', startTime='2024-01-01T09:40:00Z', endTime='2024-01-01T10:00:00Z')

# Relacionamentos
doc.wasGeneratedBy(aligned_chromatograms, align_chromatograms)
doc.wasGeneratedBy(pubchem_data, fetch_pubchem_data)
doc.wasGeneratedBy(final_report, generate_final_report)
doc.used(align_chromatograms, dataset_raw)
doc.used(fetch_pubchem_data, aligned_chromatograms)
doc.used(generate_final_report, pubchem_data)
doc.used(generate_final_report, aligned_chromatograms)
doc.wasAssociatedWith(align_chromatograms, pipeline_system)
doc.wasAssociatedWith(generate_final_report, analyst)

# Gerar e salvar a imagem do modelo PROV
dot = prov_to_dot(doc)
dot.write_png("prov_model_colab.png")
print("Imagem do modelo PROV gerada com sucesso!")

# Exibir a imagem diretamente no Colab
Image("prov_model_colab.png")
