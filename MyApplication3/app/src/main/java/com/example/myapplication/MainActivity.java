package com.example.myapplication;

import android.os.Handler;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.DataInputStream;
import java.io.DataOutputStream;

import java.io.IOException;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {

    Button connect_btn; // ip 받아오는 버튼

    EditText ip_edit; // ip 에디트
    TextView show_text; // 서버에서온거 보여주는 에디트
    // 소켓통신에 필요한것
    private String html = "";
    private Handler mHandler;

    private Socket socket;
    private WebView mWebView;
    private WebSettings mWebSettings;
    private DataOutputStream dos;
    private DataInputStream dis;

    private String ip = "192.168.0.9"; // IP 번호
    private int port = 8080; // port 번호

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        WebView mWebView = (WebView) findViewById(R.id.webview);

        mWebView.setWebViewClient(new WebViewClient());
        mWebSettings = mWebView.getSettings(); //각종 환경 설정 가능여부
        mWebSettings.setJavaScriptEnabled(true); // 자바스크립트 허용여부
        mWebSettings.setSupportMultipleWindows(false); // 윈도우 여러개 사용여부
        mWebSettings.setLayoutAlgorithm(WebSettings.LayoutAlgorithm.SINGLE_COLUMN); // 컨텐츠사이즈 맞추기
        mWebSettings.setCacheMode(WebSettings.LOAD_CACHE_ELSE_NETWORK); // 캐시 허용 여부
        mWebSettings.setUseWideViewPort(true); // wide viewport 사용 여부
        mWebSettings.setSupportZoom(true); // Zoom사용여부
        mWebSettings.setJavaScriptCanOpenWindowsAutomatically(false); // 자바스크립트가 window.open()사용할수있는지 여부
        mWebSettings.setLoadWithOverviewMode(true); // 메타태그 허용 여부
        mWebSettings.setBuiltInZoomControls(false); // 화면 확대 축소 허용 여부
        mWebSettings.setDomStorageEnabled(true); // 로컬저장소 허용 여부
        Button renew=findViewById(R.id.button);
        connect();
        renew.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d("IN","HI");
                mWebView.loadUrl("http://192.168.0.9:5000");
            }
        });


    }





    // 로그인 정보 db에 넣어주고 연결시켜야 함.
    void connect(){
        mHandler = new Handler();
        Log.w("connect","연결 하는중");
// 받아오는거
        Thread checkUpdate = new Thread() {
            public void run() {
// ip받기
                String newip = "192.168.0.9";

// 서버 접속
                try {
                    socket = new Socket(newip, port);
                    Log.w("서버 접속됨", "서버 접속됨");
                } catch (IOException e1) {
                    Log.w("서버접속못함", "서버접속못함");
                    e1.printStackTrace();
                }

                Log.w("edit 넘어가야 할 값 : ","안드로이드에서 서버로 연결요청");

                try {
                    dos = new DataOutputStream(socket.getOutputStream()); // output에 보낼꺼 넣음
                    dis = new DataInputStream(socket.getInputStream()); // input에 받을꺼 넣어짐
                    dos.writeUTF("안드로이드에서 서버로 연결요청");

                } catch (IOException e) {
                    e.printStackTrace();
                    Log.w("버퍼", "버퍼생성 잘못됨");
                }
                Log.w("버퍼","버퍼생성 잘됨");

// 서버에서 계속 받아옴 - 한번은 문자, 한번은 숫자를 읽음. 순서 맞춰줘야 함.
                try {
                    String line = "";
                    int line2;
                    while(true) {
                        line = (String)dis.readUTF();
                        line2 = (int)dis.read();
                        Log.w("서버에서 받아온 값 :",""+line);
                        Log.w("서버에서 받아온 값 ",""+line2);
                        MainActivity.this.runOnUiThread(new Runnable() {
                            public void run() {
                                Toast.makeText(MainActivity.this, "STEALING", Toast.LENGTH_SHORT).show();
                            }
                        });
                    }
                }catch (Exception e){

                }
            }
        };
// 소켓 접속 시도, 버퍼생성
        checkUpdate.start();
    }
}