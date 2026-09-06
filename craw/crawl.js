const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs-extra');
const path = require('path');

/**
 * Hàm crawl nội dung truyện
 * @param {Object} config Thông tin cấu hình crawl
 */
async function crawlNovels(config) {
    const { storyName, startChapter, endChapter, baseUrl, contentSelector, basePath } = config;

    // Đường dẫn folder đầu ra: basePath/storyName
    const outputFolder = path.join(basePath, storyName);

    try {
        // Tạo folder lưu truyện nếu chưa tồn tại
        await fs.ensureDir(outputFolder);
        console.log(`[+] Đã chuẩn bị thư mục xuất dữ liệu: ${outputFolder}`);

        for (let current = startChapter; current <= endChapter; current++) {
            // Tạo URL tương ứng cho từng chương (đảm bảo đúng định dạng gạch chéo cuối URL)
            const cleanBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
            const chapterUrl = `${cleanBaseUrl}/chuong-${current}/`;

            console.log(`[*] Đang tải chương ${current}: ${chapterUrl}...`);

            try {
                // Gửi HTTP Request lấy HTML của trang
                const { data: html } = await axios.get(chapterUrl, {
                    headers: {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    },
                    timeout: 10000 // Timeout 10s
                });

                // Load HTML vào cheerio để query DOM
                const $ = cheerio.load(html);

                // Lấy phần tử chứa nội dung
                const contentElement = $(contentSelector);

                if (!contentElement.length) {
                    console.warn(`[!] Không tìm thấy element với selector '${contentSelector}' tại chương ${current}`);
                    continue;
                }

                // Xử lý xuống dòng: Thay thế thẻ <br> bằng ký tự xuống dòng \n
                contentElement.find('br').replaceWith('\n');

                // Lấy văn bản thuần và xóa khoảng trắng dư thừa
                const chapterContent = contentElement.text().trim();

                // Đặt tên file dạng: 1.txt, 2.txt, ...
                const fileName = `${current}.txt`;
                const filePath = path.join(outputFolder, fileName);

                // Ghi nội dung vào file .txt
                await fs.writeFile(filePath, chapterContent, 'utf-8');
                console.log(`[✓] Đã lưu chương ${current} -> ${fileName}`);

            } catch (err) {
                console.error(`[X] Lỗi khi crawl chương ${current}: ${err.message}`);
            }

            // Nghỉ 1 giây giữa các lượt request để tránh bị chặn IP (Rate limit)
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        console.log(`\n=== Hoàn tất crawl truyện ${storyName}! ===`);

    } catch (error) {
        console.error(`[X] Có lỗi xảy ra trong quá trình khởi tạo:`, error.message);
    }
}

// ==========================================
// THIẾT LẬP CÁC THAM SỐ VÀ CHẠY
// ==========================================

const config = {
    storyName: '',
    startChapter: 1,
    endChapter: 100, // Thay đổi số lượng chương muốn tải tại đây
    baseUrl: 'https://truyenfull.live/thieu-gia-bi-bo-roi',
    contentSelector: '#chapter-c', // ID phần tử HTML chứa nội dung truyện trên TruyenFull
    basePath: path.join(__dirname, '../stories/truyen-001/script')
};

// Chạy hàm crawl
crawlNovels(config);