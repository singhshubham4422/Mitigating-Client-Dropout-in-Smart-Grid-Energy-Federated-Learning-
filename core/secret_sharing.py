# ============================================
# File: core/secret_sharing.py
# Purpose: Shamir Secret Sharing for mask seeds
# ============================================

import random
PRIME = 2**61 - 1

def _eval_polynomial(coeffs, x):
    result = 0
    for power, coef in enumerate(coeffs):
        result = (result + coef * pow(x, power, PRIME)) % PRIME
    return result


def split_secret(secret, num_shares, threshold):
    """
    Split secret into shares using Shamir.
    """
    coeffs = [secret] + [
        random.randrange(0, PRIME) for _ in range(threshold - 1)
    ]

    shares = []
    for i in range(1, num_shares + 1):
        shares.append((i, _eval_polynomial(coeffs, i)))

    return shares


def reconstruct_secret(shares):
    """
    Lagrange interpolation.
    """
    secret = 0

    for j, (xj, yj) in enumerate(shares):
        numerator = 1
        denominator = 1

        for m, (xm, _) in enumerate(shares):
            if m != j:
                numerator = (numerator * (-xm)) % PRIME
                denominator = (denominator * (xj - xm)) % PRIME

        lagrange = numerator * pow(denominator, -1, PRIME)
        secret = (secret + yj * lagrange) % PRIME

    return secret
