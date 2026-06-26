# 실행 한 번만 하면 됨 — gen_cert.py 파일 만들어서 실행
from OpenSSL import crypto

key = crypto.PKey()
key.generate_key(crypto.TYPE_RSA, 2048)

cert = crypto.X509()
cert.get_subject().CN = "localhost"
cert.set_serial_number(1)
cert.gmtime_adj_notBefore(0)
cert.gmtime_adj_notAfter(365*24*60*60)
cert.set_issuer(cert.get_subject())
cert.set_pubkey(key)
cert.sign(key, "sha256")

with open("../cert.pem", "wb") as f:
    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
with open("../key.pem", "wb") as f:
    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

print("인증서 생성 완료")