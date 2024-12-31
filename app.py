from flask import Flask, request, redirect, url_for, render_template
import os
import subprocess
import time
import logging
from werkzeug.utils import secure_filename
import pkg_resources

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'raw'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)
print("Início do script")
try:
    
    print("RDKit importado com sucesso")
except ImportError as e:
    print(f"Erro de importação: {e}")
except Exception as e:
    print(f"Erro inesperado: {e}")

# Configuração do log para exibir mensagens de debug no console


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        app.logger.debug("Recebendo arquivos enviados pelo usuário.")

        if 'raw_files' not in request.files or 'cqcl_file' not in request.files:
            app.logger.warning("Arquivos não encontrados na solicitação.")
            return redirect(request.url)

        raw_files = request.files.getlist('raw_files')
        cqcl_file = request.files['cqcl_file']
        cqcl_reinj_file = request.files['cqcl_reinj_file']
        cqcl_neg_file = request.files['cqcl_neg_file']

        if not all(allowed_file(f.filename) for f in raw_files) or not allowed_file(cqcl_file.filename):
            app.logger.error("Arquivo não permitido.")
            return "Arquivo não permitido."

        for file in raw_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.logger.debug(f"Arquivo {filename} salvo na pasta de uploads.")

        cqcl_filename = secure_filename(cqcl_file.filename)
        cqcl_file.save(os.path.join(app.config['UPLOAD_FOLDER'], cqcl_filename))
        app.logger.debug(f"Arquivo {cqcl_filename} salvo na pasta de uploads.")

        cqcl_reinj_filename = secure_filename(cqcl_reinj_file.filename)
        cqcl_reinj_file.save(os.path.join(app.config['UPLOAD_FOLDER'], cqcl_reinj_filename))
        app.logger.debug(f"Arquivo {cqcl_reinj_filename} salvo na pasta de uploads.")

        cqcl_neg_filename = secure_filename(cqcl_neg_file.filename)
        cqcl_neg_file.save(os.path.join(app.config['UPLOAD_FOLDER'], cqcl_neg_filename))
        app.logger.debug(f"Arquivo {cqcl_neg_filename} salvo na pasta de uploads.")

        start_time = time.time()  # Registrar o tempo de início
        app.logger.info("Iniciando processamento dos arquivos.")

        for file in raw_files:
           
                sample_filename = secure_filename(file.filename)
                sample_filepath = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
                cqcl_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cqcl_filename)
                cqcl_reinj_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cqcl_reinj_filename)
                cqcl_neg_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cqcl_neg_filename)

                app.logger.debug(f"Processando arquivo {sample_filename} com o script dopinhoReduzido.py.")
                
                try:
                    result = subprocess.run(
                        ['python', 'dopinhoReduzido.py', sample_filepath, cqcl_filepath, cqcl_reinj_filepath, cqcl_neg_filepath],
                        capture_output=True,
                        text=True
                    )
                    
                    # Captura a saída e o erro padrão
                    app.logger.debug(f"Resultado do processamento de {sample_filename}: {result.stdout}")
                    
                    if result.stderr:
                        app.logger.error(f"Erro ao processar {sample_filename}: {result.stderr}")

                except Exception as e:
                    app.logger.error(f"Erro ao executar dopinhoReduzido.py: {e}")


        end_time = time.time()  # Registrar o tempo de término
        execution_time = end_time - start_time  # Calcular o tempo de execução
        app.logger.info(f"Processamento concluído em {execution_time:.2f} segundos.")

        # Renderizar o template passando o execution_time
        return render_template('index.html', execution_time=execution_time, result=result.stdout)

    return render_template('index.html')

if __name__ == '__main__':
    # Habilita o modo de debug do Flask
    app.run(debug=True, port=5001)
print("Script principal iniciado")


# Lista os nomes dos pacotes que você está usando
libraries = [
    "flask",
    "werkzeug",
]

# Imprime o nome e a versão de cada biblioteca
for lib in libraries:
    try:
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib}: Não está instalado")
