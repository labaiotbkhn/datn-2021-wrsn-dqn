# Cài đặt môi trường

pip install -r requirements.txt

# Chạy chương trình

Chạy file main.py theo lệnh sau:

```bash
python main.py --filedata data/thaydoitileguitin.csv --type thaydoitileguitin --maxTime 90000
```

filedata: đường dẫn tới file dữ liệu tạo ra (trong folder data)

type: các loại thí nghiệm. Có 3 loại thí nghiệm: thaydoitileguitin(thay đổi tỉ lệ gửi gói tin), thaydoitarget(thay đổi số lượng mục tiêu theo dõi), thaydoinode(Thay đổi số lượng nút cảm biến)

maxTime: Thời gian để chạy thí nghiệm xem số lượng nút cảm biến và số lượng mục tiêu theo dõi trong một lượng thời gian. Mặc định không cần thêm
