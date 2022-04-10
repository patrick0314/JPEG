# JPEG

## Framework

![](https://i.imgur.com/PWbTtZt.png)

將 JPEG 分成 4 個部分，分別是 [Preprocess](https://github.com/patrick0314/JPEG/blob/main/Preprocess.py)、[DCT](https://github.com/patrick0314/JPEG/blob/main/DCT.py)、[Quantization](https://github.com/patrick0314/JPEG/blob/main/Quantize.py)、[Huffman Coding](https://github.com/patrick0314/JPEG/blob/main/Huffman.py)。

* **Preprocess**

將圖片轉為 YCbCr 之後，使用 4:2:0 sampling method 來做取樣。

* **DCT**
    ![](https://i.imgur.com/tBSLWmJ.jpg)

* **Quantization**
    ![](https://i.imgur.com/IhkbJ1b.png)
    ![](https://i.imgur.com/7UWKsSb.png)
    ![](https://i.imgur.com/10YzNsp.png)

* **Huffman Coding**
    * 使用 Table for luminance DC coefficient differences 來做 Huffman coding 的結果：
    
    ![](https://i.imgur.com/tVCGMqi.jpg)

    
    * 使用 Table for luminance AC coefficient 來做 Huffman coding 的結果：

    ![](https://i.imgur.com/fYwyq1Z.jpg)



## Performance

一般影像的 bpp 基礎為 24，從下圖中可以發現，都有明顯幅度的壓縮成效。也可以從 PSNR 中看出還原成果是很好的。

![](https://i.imgur.com/49eXxU9.jpg)


另外，可以從 Quantization 中的 table coefficient $\tau$ 來調整 bpp 和 PSNR 的結果：

![](https://i.imgur.com/ofmvcoA.png)

從上圖可以看出，在 <img src="https://render.githubusercontent.com/render/math?math=\tau > 1">$\tau > 1$ 的時候，bpp 會下降，PSNR 則會上升；反之，在 $\tau < 1$ 的時候，bpp 會上升，PSNR 則會下降。

## Usage

JPEG 編碼：

* 將 image 放到 `./pics` 的資料夾
* 運行 `JPEGEncoding.py`
* 壓縮後的結果會放在 `./encode` 的資料夾

JPEG 解碼：

* 確認壓縮後的結果放在 `./encode` 的資料夾
* 運行 `JPEGDecoding.py`
* 解壓縮後的結果會放在 `./decode` 的資料夾

# Example

[video](https://www.youtube.com/watch?v=t0olBLgmxXc&ab_channel=0314Patrick)





###### tags: `Github` `Image Compression`
