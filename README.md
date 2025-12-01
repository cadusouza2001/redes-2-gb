# Simulação de Transmissão Digital (BPSK e QPSK)

Projeto educacional em Python que demonstra todas as etapas de um enlace digital:
conversão de texto em bits, codificação Manchester, modulação (BPSK e QPSK),
canal AWGN, detecção, decodificação e cálculo de BER.

## Requisitos
- Python 3.10+
- numpy
- matplotlib

## Como executar
1. Instale as dependências (se necessário):
   ```bash
   pip install numpy matplotlib
   ```
2. Rode a simulação padrão com sweeping de SNR (-5 dB a 15 dB):
   ```bash
   python digital_comm.py
   ```

A execução imprime logs no console com o número de erros e BER para cada SNR e
modulação, e gera o gráfico `ber_vs_snr.png` na raiz do projeto.

## Organização do código
- `BitStreamConverter`: converte texto ASCII para bits (via `numpy.unpackbits`) e
  bits de volta para texto.
- `ManchesterEncoder`: aplica a codificação Manchester (0 → [0, 1], 1 → [1, 0])
  e sua decodificação inversa.
- `BPSKModem` e `QPSKModem`: moduladores/demoduladores baseados em distância
  euclidiana mínima. O QPSK usa mapeamento Gray normalizado por \(1/\sqrt{2}\).
- `AWGNChannel`: adiciona ruído branco gaussiano aditivo a partir do SNR em dB.
- `DigitalTransmissionSimulator`: orquestra uma transmissão completa, calcula
  BER e gera o gráfico comparativo.

## Observações teóricas
- **Eficiência espectral**: QPSK transmite 2 bits por símbolo, dobrando a
  eficiência espectral em relação ao BPSK para a mesma largura de banda.
- **Robustez a ruído**: BPSK possui maior distância mínima entre símbolos,
  apresentando BER levemente menor que QPSK para a mesma relação Eb/N0.
- **Codificação Manchester**: dobra a taxa de símbolos, mas fornece componente
  DC nula e facilita a sincronização de relógio.

O código está amplamente comentado para destacar o cálculo de variância do
ruído, mapeamento de constelações e etapas de codificação/decodificação.
