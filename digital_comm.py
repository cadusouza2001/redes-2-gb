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
        return byte_array.tobytes().decode("ascii", errors="ignore")


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

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 2 != 0:
            # Ignora o último bit se a contagem for ímpar para manter pares.
            bits = bits[:-1]
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([self.constellation[tuple(b)] for b in bit_pairs], dtype=np.complex128)
        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        # Calcula distâncias euclidianas para cada ponto da constelação.
        bits_out = []
        for s in symbols:
            distances = np.abs(s - self.symbol_points) ** 2
            idx = np.argmin(distances)
            bits_out.extend(self.symbol_bits[idx])
        return np.array(bits_out, dtype=np.int8)


class AWGNChannel:
    """Canal AWGN com controle de SNR em dB."""

    def transmit(self, signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        # Variância complexa: cada componente (real e imaginário) terá sigma^2 = signal_power/(2*SNR).
        noise_variance = signal_power / (2 * snr_linear)
        noise_std = np.sqrt(noise_variance)
        noise = noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
        return signal + noise


class DigitalTransmissionSimulator:
    """Simulador completo com codificação Manchester, modulação e BER."""

    def __init__(self, text: str, snr_range: list[int] | np.ndarray, seed: int = 0) -> None:
        self.text = text
        self.snr_range = np.array(snr_range)
        self.rng = np.random.default_rng(seed)
        self.converter = BitStreamConverter()
        self.encoder = ManchesterEncoder()
        self.channel = AWGNChannel()
        self.modems = {
            "BPSK": BPSKModem(),
            "QPSK": QPSKModem(),
        }

    def _transmit_once(self, modem_name: str, snr_db: float) -> tuple[int, int, str]:
        bits = self.converter.text_to_bits(self.text)
        encoded = self.encoder.encode(bits)
        symbols = self.modems[modem_name].modulate(encoded)
        noisy = self.channel.transmit(symbols, snr_db, self.rng)
        detected = self.modems[modem_name].demodulate(noisy)
        decoded = self.encoder.decode(detected)
        min_len = min(len(bits), len(decoded))
        bit_errors = int(np.sum(bits[:min_len] != decoded[:min_len]))
        recovered_text = self.converter.bits_to_text(decoded)
        return bit_errors, min_len, recovered_text

    def run(self) -> dict[str, list[float]]:
        ber_results: dict[str, list[float]] = {name: [] for name in self.modems}
        for snr_db in self.snr_range:
            for modem_name in self.modems:
                errors, total, _ = self._transmit_once(modem_name, snr_db)
                ber = errors / total if total else 0.0
                ber_results[modem_name].append(ber)
                print(
                    f"SNR: {snr_db}dB | Modulação: {modem_name} | Erros: {errors} | BER: {ber:.6f}"
                )
        return ber_results

    def plot(self, ber_results: dict[str, list[float]], output_file: str = "ber_vs_snr.png") -> None:
        plt.figure(figsize=(8, 5))
        for modem_name, bers in ber_results.items():
            plt.semilogy(self.snr_range, bers, marker="o", label=modem_name)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title("BER x SNR para BPSK e QPSK com codificação Manchester")
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
    snr_values = np.arange(-5, 16, 2)  # -5 dB até 15 dB em passos de 2 dB.
    simulator = DigitalTransmissionSimulator(long_text, snr_values, seed=42)
    ber = simulator.run()
    simulator.plot(ber)
