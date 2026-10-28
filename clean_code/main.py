# kode awal

# def f(x, y):
#     t = 0
#     for i in x:
#         t += i["p"] * i["q"]

#     if y == True:
#         t = t - t * 0.1

#     print(f"Total bayar: Rp{t:,.0f}")


# f(
#     [
#         {"p": 20000, "q": 2},
#         {"p": 15000, "q": 1},
#     ],
#     True,
# )

# Ganti f, x, y, t, i, p, dan q dengan nama yang jelas.
# Ubah if y == True menjadi bentuk yang lebih sederhana.
# Beri nama untuk angka diskon 0.1.
# Pisahkan fungsi menghitung total bayar dari perintah mencetak hasil. Fungsi perhitungan harus memakai return.


def hitung_total_bayar(daftar_item, member):
    total_harga = 0
    member_diskon = 0.1
    
    for item in daftar_item:
        total_harga += item["harga"] * item["quantitas"]
        
    if member:
        total_harga -= total_harga * member_diskon
        
    return total_harga


def format_bayar(total_bayar):
    return f"Total bayar: Rp{total_bayar:,.0f}"

def main():
    daftar_item = [
        {"harga": 20000, "quantitas": 2},
        {"harga": 15000, "quantitas": 1},
    ]
    member = True
    total_bayar = hitung_total_bayar(daftar_item, member)
    print(format_bayar(total_bayar))
    
if __name__== "__main__":
    main()