"""
Simulação educativa de um sistema de transmissão digital.

Fluxo principal:
- Converte texto ASCII em bits.
- Codifica em Manchester.
- Modula (BPSK ou QPSK).
- Canal AWGN.
- Demodula e decodifica o sinal.
- Recupera texto e calcula a BER.

Somente numpy e matplotlib são utilizados para manter o foco na matemática do
sinal e evitar bibliotecas "caixa preta".
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


class BitStreamConverter:
    """Conversão entre texto ASCII e sequências de bits."""

    @staticmethod
    def text_to_bits(text: str) -> np.ndarray:
        bytes_view = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        # np.unpackbits gera um vetor de 0s e 1s para cada byte.
        bits = np.unpackbits(bytes_view)
        return bits.astype(np.int8)

    @staticmethod
    def bits_to_text(bits: np.ndarray) -> str:
        # Garante múltiplos de 8 bits para remontar bytes.
        trimmed = bits[: len(bits) - (len(bits) % 8)]
        byte_array = np.packbits(trimmed.astype(np.uint8))
        return byte_array.tobytes().decode("ascii", errors="replace")


class ManchesterEncoder:
    """Codificação e decodificação Manchester."""

    def __init__(self, low_high: tuple[int, int] = (0, 1)) -> None:
        self.low, self.high = low_high

    def encode(self, bits: np.ndarray) -> np.ndarray:
        # Cada bit vira um par: 0 -> [low, high], 1 -> [high, low].
        encoded = np.empty(bits.size * 2, dtype=np.int8)
        encoded[0::2] = np.where(bits == 0, self.low, self.high)
        encoded[1::2] = np.where(bits == 0, self.high, self.low)
        return encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        if encoded.size % 2 != 0:
            raise ValueError("Sequência Manchester deve ter tamanho par.")
        pairs = encoded.reshape(-1, 2)
        decoded = np.where(
            (pairs[:, 0] == self.low) & (pairs[:, 1] == self.high),
            0,
            1,
        )
        return decoded.astype(np.int8)


class BPSKModem:
    """Modulador/Demodulador BPSK com mapeamento ±1."""

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        # 0 -> -1, 1 -> +1. Resultado é real, mas usamos complexo para canal.
        symbols = 2 * bits - 1
        return symbols.astype(np.complex128)

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        # Detecção por limiar (distância ao eixo real).
        return (symbols.real >= 0).astype(np.int8)

    bits_per_symbol = 1


class QPSKModem:
    """Modulador/Demodulador QPSK com mapeamento Gray."""

    def __init__(self) -> None:
        norm = 1 / np.sqrt(2)
        self.constellation = {
            (0, 0): norm * (1 + 1j),
            (0, 1): norm * (-1 + 1j),
            (1, 1): norm * (-1 - 1j),
            (1, 0): norm * (1 - 1j),
        }
        # Vetor de símbolos para decisão por distância euclidiana.
        self.symbol_points = np.array(list(self.constellation.values()))
        self.symbol_bits = np.array(list(self.constellation.keys()))
        self.bits_per_symbol = 2

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 2 != 0:
            bits = np.pad(bits, (0, 1), constant_values=0)
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([self.constellation[tuple(b)] for b in bit_pairs], dtype=np.complex128)
        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        # Calcula distâncias euclidianas vetorizadas para cada ponto da constelação.
        distances = np.abs(symbols[:, None] - self.symbol_points[None, :]) ** 2
        closest = np.argmin(distances, axis=1)
        bits_out = self.symbol_bits[closest].reshape(-1)
        return bits_out.astype(np.int8)


class AWGNChannel:
    """Canal AWGN com controle de Eb/N0 em dB."""

    def transmit(
        self, signal: np.ndarray, ebn0_db: float, bits_per_symbol: int, rng: np.random.Generator
    ) -> np.ndarray:
        signal_power = np.mean(np.abs(signal) ** 2)
        ebn0_linear = 10 ** (ebn0_db / 10)
        # Es = Eb * k -> N0 = Eb / (Eb/N0). Ruído por dimensão real: N0/2.
        noise_variance = signal_power / (2 * bits_per_symbol * ebn0_linear)
        noise_std = np.sqrt(noise_variance)
        noise = noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
        return signal + noise


class DigitalTransmissionSimulator:
    """Simulador completo com codificação Manchester, modulação e BER."""

    def __init__(self, text: str, ebn0_range: list[int] | np.ndarray, seed: int = 0) -> None:
        self.text = text
        self.ebn0_range = np.array(ebn0_range)
        self.rng = np.random.default_rng(seed)
        self.converter = BitStreamConverter()
        self.encoder = ManchesterEncoder()
        self.channel = AWGNChannel()
        self.modems = {
            "BPSK": BPSKModem(),
            "QPSK": QPSKModem(),
        }

    def _transmit_once(self, modem_name: str, ebn0_db: float) -> tuple[int, int, str]:
        bits = self.converter.text_to_bits(self.text)
        encoded = self.encoder.encode(bits)
        modem = self.modems[modem_name]
        symbols = modem.modulate(encoded)
        noisy = self.channel.transmit(symbols, ebn0_db, modem.bits_per_symbol, self.rng)
        detected = self.modems[modem_name].demodulate(noisy)
        decoded = self.encoder.decode(detected)
        min_len = min(len(bits), len(decoded))
        bit_errors = int(np.sum(bits[:min_len] != decoded[:min_len]))
        recovered_text = self.converter.bits_to_text(decoded)
        return bit_errors, min_len, recovered_text

    def run(self) -> dict[str, list[float]]:
        # Configuração Inicial
        print("\n" + "="*60)
        print("   SIMULAÇÃO DE SISTEMA DE TRANSMISSÃO DIGITAL (BPSK/QPSK)")
        print("="*60)
        print(f"Texto original carregado ({len(self.text)} caracteres).")
        print("Configurando parâmetros de simulação:")
        print("  - Modulações: BPSK, QPSK")
        print("  - Codificação: Manchester")
        print(f"  - Range Eb/N0: {self.ebn0_range[0]} dB a {self.ebn0_range[-1]} dB")
        
        # PAUSA 1
        input("\n>>> Pressione ENTER para iniciar a varredura de BER e gerar gráficos...")
        
        print("\nExecutando varredura de BER...")
        ber_results: dict[str, list[float]] = {name: [] for name in self.modems}
        for ebn0_db in self.ebn0_range:
            for modem_name in self.modems:
                errors, total, _ = self._transmit_once(modem_name, ebn0_db)
                ber = errors / total if total else 0.0
                ber_results[modem_name].append(ber)
                print(
                    f"Eb/N0: {ebn0_db:3d} dB | Modulação: {modem_name} | Erros: {errors:4d} | BER: {ber:.6f}"
                )
        return ber_results

    def plot(self, ber_results: dict[str, list[float]], output_file: str = "ber_vs_ebn0.png") -> None:
        plt.figure(figsize=(8, 5))
        for modem_name, bers in ber_results.items():
            plt.semilogy(self.ebn0_range, bers, marker="o", linestyle="None", label=f"{modem_name} (sim)")

        # Curvas teóricas
        ebn0_linear = 10 ** (self.ebn0_range / 10)
        theoretical_ber = 0.5 * erfc(np.sqrt(ebn0_linear))
        plt.semilogy(self.ebn0_range, theoretical_ber, label="BPSK/QPSK (teórico)")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("BER")
        plt.title("BER x Eb/N0 para BPSK e QPSK com codificação Manchester")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Gráfico salvo em {output_file}")


if __name__ == "__main__":
    long_text = (
        "Engenharia de Telecomunicacoes exige clareza, precisao e muita pratica para"
        " dominar os fundamentos de sistemas digitais. "
        "Esta simulacao em Python demonstra o impacto do ruido sobre diferentes"
        " modulacoes com codificacao Manchester."
    )
    
    # 1. Range estendido (-10 dB a 16 dB)
    ebn0_values = np.arange(-10, 16, 2)
    
    simulator = DigitalTransmissionSimulator(long_text, ebn0_values, seed=42)
    
    # Executa a simulação principal e plota
    ber = simulator.run()
    simulator.plot(ber)

    # PAUSA 2
    input("\n>>> Pressione ENTER para analisar a Eficiência Espectral...")

    # 2. Eficiência Espectral
    manchester_rate = 0.5 
    efficiencies = {
        "BPSK": 1 * manchester_rate,
        "QPSK": 2 * manchester_rate,
    }
    print("\n" + "="*30)
    print("ANÁLISE DE EFICIÊNCIA ESPECTRAL")
    print("="*30)
    print("(Considerando overhead da codificação Manchester)")
    for name, eff in efficiencies.items():
        print(f" - {name}: {eff:.2f} bits/s/Hz")

    # PAUSA 3
    input("\n>>> Pressione ENTER para iniciar a Demonstração Prática (Recuperação de Texto)...")

    # 3. Demonstração Prática
    demonstration_snrs = [0, 10] 
    
    print("\n" + "="*60)
    print("   DEMONSTRAÇÃO PRÁTICA: IMPACTO DO RUÍDO NO TEXTO")
    print("="*60)

    for i, snr in enumerate(demonstration_snrs):
        if i > 0:
             # PAUSA 4 (Entre os cenários de teste, opcional, para dar suspense)
             input(f"\n>>> Pressione ENTER para testar o próximo cenário (Eb/N0 = {snr} dB)...")
        
        print(f"\n--- Cenário {i+1}: Eb/N0 = {snr} dB ---")
        for modem_name in simulator.modems:
            errors, total, recovered = simulator._transmit_once(modem_name, snr)
            ber_val = errors / total
            print(f"\n[{modem_name}] BER: {ber_val:.6f} (Erros: {errors}/{total})")
            print("Texto Recuperado:")
            print(f"{recovered}")
            print("-" * 60)
            
    print("\nFim da execução.")