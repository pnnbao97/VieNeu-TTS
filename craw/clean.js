const fs = require('fs-extra');
const path = require('path');

// 1. Đường dẫn đến thư mục chứa các file .txt
const STORIES_DIR = path.join(__dirname, '../stories/truyen-001/script');

// 2. Danh sách các cụm từ không cần thiết cần xóa
const JUNK_TEXTS = [
    'truyen full',
    'truyenfullvn',
    'truyenfulllivetruyenfull',
    'truyenfull vn',
    'truyenfull' // Đặt 'truyenfull' ở cuối để tránh xóa đè các từ dài hơn ở trên
];

async function cleanTextFiles() {
    try {
        console.log(`[*] Đang quét thư mục: ${STORIES_DIR}`);

        // Kiểm tra thư mục có tồn tại không
        if (!await fs.pathExists(STORIES_DIR)) {
            console.error(`[X] Thư mục không tồn tại: ${STORIES_DIR}`);
            return;
        }

        // Lấy danh sách tất cả các file trong thư mục
        const files = await fs.readdir(STORIES_DIR);
        const txtFiles = files.filter(file => file.endsWith('.txt'));

        if (txtFiles.length === 0) {
            console.log('[!] Không tìm thấy file .txt nào trong thư mục.');
            return;
        }

        console.log(`[*] Phát hiện ${txtFiles.length} file .txt. Bắt đầu dọn dẹp...\n`);

        // Tạo Regex để tìm tất cả cụm từ rác (không phân biệt hoa/thường - 'gi')
        // Escaping các ký tự đặc biệt nếu có
        const escapedJunk = JUNK_TEXTS.map(text => text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&'));
        const regex = new RegExp(escapedJunk.join('|'), 'gi');

        let updatedCount = 0;

        for (const file of txtFiles) {
            const filePath = path.join(STORIES_DIR, file);

            // Đọc nội dung file
            let content = await fs.readFile(filePath, 'utf-8');

            // Xóa text rác
            const cleanedContent = content.replace(regex, '');

            // Nếu nội dung có sự thay đổi thì ghi đè lại file
            if (content !== cleanedContent) {
                // Xóa bớt khoảng trắng hoặc dòng trống thừa ở cuối file
                const finalContent = cleanedContent.trim();
                
                await fs.writeFile(filePath, finalContent, 'utf-8');
                updatedCount++;
                console.log(`[✓] Đã dọn dẹp: ${file}`);
            }
        }

        console.log(`\n=== HOÀN TẤT: Đã xử lý ${updatedCount}/${txtFiles.length} file! ===`);

    } catch (error) {
        console.error('[X] Có lỗi xảy ra:', error.message);
    }
}

// Chạy script
cleanTextFiles();