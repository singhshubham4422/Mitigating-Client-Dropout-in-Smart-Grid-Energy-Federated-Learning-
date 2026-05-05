# ============================================
# File: core/crypto_utils.py
# Purpose: Real ECDH key exchange for SMPC
# ============================================

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def generate_keypair():
    """Generate ECDH private/public key pair."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

def derive_shared_key(private_key, peer_public_key):
    """Derive a shared secret using ECDH + HKDF."""
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"federated-secure-aggregation",
    ).derive(shared_secret)

    return derived_key
