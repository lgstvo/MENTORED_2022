#configurações do script
origem_pacotes = "../data/CIC/pcaps"
destino_csvs = "../data/CIC/csvs/"
pcap_key = "###PCAP###"
csv_out_key = "###OUT###"
comando = "tshark -o gui.column.format:\"SP,%uS,DP,%uD,SA,%s,DA,%d,PR,%p\" -r "+pcap_key+" -T fields -e frame.time -e frame.time_epoch -e frame.len -e _ws.col.SA -e _ws.col.DA -e _ws.col.PR -e _ws.col.SP -e _ws.col.DP -E header=y -E separator=\";\" >"+csv_out_key+".csv"
num_processos = 3
lista_comandos = []

#DEFINIR IMPORTS
import os
import multiprocessing



#MÉTODO PARA LER OS PACOTES DE UMA PASTA
def ler_pacotes_pasta():
    return sorted(os.listdir(origem_pacotes))


def executar_comando(executar):
    stream = os.popen(executar)
    #return int(str(stream.read()).replace("\n", ""))
    return str(stream.read()).replace("\n", "")

pacotes = ler_pacotes_pasta()

for pacote in pacotes:
    lista_comandos.append(comando.replace(pcap_key,origem_pacotes+"/"+pacote).replace(csv_out_key,destino_csvs+pacote))

print(lista_comandos[0])
    
pool = multiprocessing.Pool(processes=num_processos)
pool.map(executar_comando,lista_comandos)