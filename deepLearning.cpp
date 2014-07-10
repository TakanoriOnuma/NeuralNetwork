#include <time.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#define DEEP_LEARNING 1         // 深層学習を行うか（学習を相当やるか）

using namespace std;


const int    N             = 16;        // sin波のプロットする個数
const int    NUM_STEP      = 100;       // 学習させる回数
const int    DEEP_STEP     = 1000;   // 深層学習で1ステップで学習させる回数
const int    DEEP_NUM_STEP = 1000;     // 深層学習で学習させる回数
const int    Patterns      = 50;
const double ETA           = 0.01;
const double Alpha         = 0.8;
const double PAI           = 3.14159265359;
const double ErrorEv       = 0.01;
const double Rlow          = -1.0;
const double Rhigh         = 1.0;
const double OmegaLow      = PAI;
const double OmegaHigh     = 4 * PAI;

double Eta = ETA;

inline double fout(double x)
{
    return tanh(x);
}

inline double delta_fout(double fout_value)
{
    return 1 - fout_value * fout_value;
}

// dec_point_num <= 3がいい。4から少し動作が変
double my_rand(double min, double max, int dec_point_num)
{
    double value = 1.0;
    for(int i = 0; i < dec_point_num; i++){
        value *= 10.0;
    }
    return (double)(rand() % (int)((max - min) * value + 1)) / value + min;
}

class Neuron {
    double u;
    double x;
    vector<double> w;
    vector<double> delta_w;
    double bias;
    double delta_bias;

public:
    Neuron(const vector<double>& w, double bias)
        : w(w), bias(bias), u(0.0), x(0.0), delta_w(0), delta_bias(0.0)
    {
        delta_w.resize(w.size());
        for(int i = 0; i < delta_w.size(); i++) {
            delta_w[i] = 0.0;
        }
    }

    // --- getter --- //
    double getU() const { return u; }
    double getX() const { return x; }
    const vector<double>& getW() const { return w; }
    const vector<double>& getDeltaW() const { return delta_w; }
    double getBias() const { return bias; }
    double getDeltaBias() const { return delta_bias; }

    // --- setter --- //
    void setW(const vector<double>& w) { this->w = w; }
    void setDeltaW(const vector<double>& delta_w) { this->delta_w = delta_w; }
    void setBias(double bias) { this->bias = bias; }
    void setDeltaBias(double delta_bias) { this->delta_bias = delta_bias; }

    void scalarProduct(const vector<double>& in) {
        if(w.size() == in.size()) {
            u = 0.0;
            for(int i = 0; i < w.size(); i++) {
                u += w[i] * in[i];
            }
            x = fout(u + bias);
        }
    }

    // 入力層の入力信号はそのまま入れる
    void setX(double x) { this->x = x; }
};

void forwardPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& inp)
{
    for(int i = 0; i < neurons[0].size(); i++) {
        neurons[0][i]->setX(inp[i]);
    }
    for(int i = 1; i < neurons.size(); i++) {
        vector<double> inp(neurons[i - 1].size());
        for(int j = 0; j < inp.size(); j++) {
            inp[j] = neurons[i - 1][j]->getX();
        }
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j]->scalarProduct(inp);
        }
    }
}

void backPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& tsignal)
{

    vector<Neuron*>& out_neurons = neurons[neurons.size() - 1];     // 出力層ニューロン
    // 入力信号と教師信号との誤差を求める
    vector<double> dwo(out_neurons.size());
    for(int i = 0; i < dwo.size(); i++) {
        dwo[i] = (tsignal[i] - out_neurons[i]->getX()) * delta_fout(out_neurons[i]->getX());
    }


    // 出力層の結合荷重値を変更する
    for(int i = 0; i < out_neurons.size(); i++) {
        vector<double> w       = out_neurons[i]->getW();
        vector<double> delta_w = out_neurons[i]->getDeltaW();
        for(int j = 0; j < w.size(); j++) {
            delta_w[j] = Eta * dwo[i] * neurons[neurons.size() - 2][j]->getX() + Alpha * delta_w[j];
            w[j] += delta_w[j];
        }
        out_neurons[i]->setW(w);
        out_neurons[i]->setDeltaW(delta_w);
    }

    // 出力層のしきい値を変更する
    for(int i = 0; i < out_neurons.size(); i++) {
        double bias = out_neurons[i]->getBias();
        double delta_bias = out_neurons[i]->getDeltaBias();
        delta_bias = Eta * dwo[i] + Alpha * delta_bias;
        bias += delta_bias;
        out_neurons[i]->setBias(bias);
        out_neurons[i]->setDeltaBias(delta_bias);
    }

    // 中間層の結合荷重値としきい値を変更する
    vector<double> up_delta  = dwo;
    for(int i = (int)neurons.size() - 2; i >= 1; i--) {
        vector<double> now_delta(neurons[i].size());

        // 誤差伝搬で誤差を求める
        for(int j = 0; j < neurons[i].size(); j++) {
            double sum = 0.0;
            
            for(int k = 0; k < neurons[i + 1].size(); k++) {
                sum += up_delta[k] * neurons[i + 1][k]->getW()[j];
            }
            now_delta[j] = delta_fout(neurons[i][j]->getX()) * sum;
        }

        // 現在の層の結合荷重値を変更する
        for(int j = 0; j < neurons[i].size(); j++) {
            vector<double> w = neurons[i][j]->getW();
            vector<double> delta_w = neurons[i][j]->getDeltaW();
            for(int k = 0; k < w.size(); k++) {
                delta_w[k] = Eta * now_delta[j] * neurons[i - 1][k]->getX() + Alpha * delta_w[k];
                w[k] += delta_w[k];
            }
            neurons[i][j]->setW(w);
            neurons[i][j]->setDeltaW(delta_w);
        }

        // 現在の層のしきい値を変更する
        for(int j = 0; j < neurons[i].size(); j++) {
            double bias = neurons[i][j]->getBias();
            double delta_bias = neurons[i][j]->getDeltaBias();
            delta_bias = Eta * now_delta[j] + Alpha * delta_bias;
            bias += delta_bias;
            neurons[i][j]->setBias(bias);
            neurons[i][j]->setDeltaBias(delta_bias);
        }

        up_delta = now_delta;
    }
}

double calcSignalError(vector<vector<Neuron*>>& neurons, const vector<double> inp_dats[], const vector<double> tsignal[], const int patterns)
{
    double error = 0.0;
    const vector<Neuron*>& out_neurons = neurons[neurons.size() - 1];

    // 一連の学習データを繰り返して、誤差を集計する
    for(int i = 0; i < patterns; i++) {
        forwardPropagation(neurons, inp_dats[i]);

        for(int j = 0; j < out_neurons.size(); j++) {
            error += pow(tsignal[i][j] - out_neurons[j]->getX(), 2.0);
        }
    }
    error *= 0.5;

    return error;
}

double calcGeneralError(vector<vector<Neuron*>>& neurons)
{
    double error = 0.0;
    const vector<Neuron*>& out_neurons = neurons[neurons.size() - 1];
    vector<double> inp_dats(out_neurons.size());

    // グラフで使う値を使って汎用誤差を集計する
    for(double A = 0.1; A <= 0.8; A += 0.02) {
        for(double Omega = OmegaLow; Omega <= OmegaHigh; Omega += 0.1) {
            for(int i = 0; i < N + 1; i++) {
                double x = 2.0 * i / N - 1.0;
                inp_dats[i] = A * sin(Omega * x);
            }

            forwardPropagation(neurons, inp_dats);

            for(int i = 0; i < out_neurons.size(); i++) {
                error += pow(inp_dats[i] - out_neurons[i]->getX(), 2.0);
            }
        }
    }
    error *= 0.5;

    return error;
}

void outNetworkProperty(const char* filename, const vector<vector<Neuron*>>& neurons)
{
    ofstream ofs(filename);

    ofs << "N:"       << N        << endl;
    ofs << "Eta:"     << Eta      << endl;
    ofs << "Alpha:"   << Alpha    << endl;
    ofs << "ErrorEv:" << ErrorEv  << endl;
    ofs << "Rlow:"    << Rlow     << endl;
    ofs << "Rhigh:"   << Rhigh    << endl;
    ofs << "NumStep:" << NUM_STEP << endl;
    ofs << "--- Neuron 階層:" << neurons.size() << " ---" << endl;
    for(int i = 0; i < neurons.size(); i++) {
        ofs << "第" << (i + 1) << "層:" << "ニューロン" << neurons[i].size() << "個" << endl;
    }
}

// 中間層の出力結果をファイルに出力
void outMiddleData(const char* filename, vector<vector<Neuron*>>& neurons)
{
    ofstream ofs_middle(filename);  // 中間層のデータ
    ofs_middle << "# A" << "\t" << "ω" << "\t";
    for(int j = 0; j < neurons[neurons.size() / 2].size(); j++) {
        ofs_middle << "neuron[" << neurons.size() / 2 << "][" << j << "]" << "\t";
    }
    ofs_middle << endl;

    // 中間層の出力を求める
    for(double Omega = OmegaLow; Omega <= OmegaHigh; Omega += 0.1) {
        ofs_middle << "# ω=" << Omega << endl;
        for(double A = 0.1; A <= 0.8; A += 0.02) {
            // 入力するsin波を作る
            vector<double> sin_dat;
            for(int i = 0; i < N + 1; i++) {
                double in_data = 2.0 * i / N - 1.0;
                sin_dat.push_back(A * sin(Omega * in_data));
            }
        
            forwardPropagation(neurons, sin_dat);

            // 中間層の出力をファイルに出力
            ofs_middle << A << "\t" << Omega << "\t";
            const vector<Neuron*>& mid_neurons = neurons[neurons.size() / 2];
            for(int i = 0; i < mid_neurons.size(); i++) {
                ofs_middle << mid_neurons[i]->getX() << "\t";
            }
            ofs_middle << endl;
        }
        ofs_middle << endl;
    }
}


// 学習後のsinデータをファイルに出力
void outLearningSinData(const char* filename, vector<vector<Neuron*>> neurons, const vector<double> inp_dats[], const vector<double> tsignal[])
{
    // 結果を出力
    ofstream ofs_sin(filename);
    ofs_sin << "# x" << "\t";
    for(int i = 0; i < Patterns; i++) {
        ofs_sin << "tsignal[" << i << "]\t" << "output[" << i << "]\t";
    }
    ofs_sin << endl;

    double out_sin[Patterns][N + 1];        // sinの学習結果

    // 学習後のsinを求める
    for(int i = 0; i < Patterns; i++) {
        forwardPropagation(neurons, inp_dats[i]);
        for(int j = 0; j < neurons[neurons.size() - 1].size(); j++) {
            out_sin[i][j] = neurons[neurons.size() - 1][j]->getX();
        }
    }

    for(int i = 0; i < N + 1; i++) {
        ofs_sin << (2.0 * i / N - 1.0) << "\t";
        for(int j = 0; j < Patterns; j++) {
            ofs_sin << tsignal[j][i] << "\t" << out_sin[j][i] << "\t";
        }
        ofs_sin << endl;
    }

}

// 一般のsinデータでの出力結果をファイルに出力
void outGeneralSinData(const char* filename, vector<vector<Neuron*>> neurons)
{
    ofstream ofs_sin(filename);
    ofs_sin << "# x" << "\t";
    for(double A = 0.1; A <= 0.8; A += 0.1) {
        for(double Omega = OmegaLow; Omega <= OmegaHigh; Omega += 0.5) {
            ofs_sin << "A = " << A << ", Omega = " << Omega << "\t" << "output" << "\t";
        }
    }
    ofs_sin << endl;

    const int pattern_A = (0.8 - 0.1) / 0.1 + 1;
    const int pattern_Omega = (OmegaHigh - OmegaLow) / 0.5 + 1;
    // メモリの確保
    double** out_sin = new double* [2 * pattern_A * pattern_Omega];
    for(int i = 0; i < 2 * pattern_A * pattern_Omega; i++) {
        out_sin[i] = new double[N + 1];
    }
    
    // 出力結果をメモリに保存する
    int idx = 0;       // 配列の添え字番号
    for(double A = 0.1; A <= 0.8; A += 0.1) {
        for(double Omega = OmegaLow; Omega <= OmegaHigh; Omega += 0.5) {
            vector<double> inp_dats(N + 1);
            for(int i = 0; i < N + 1; i++) {
                double x = 2.0 * i / N - 1.0;
                inp_dats[i] = A * sin(Omega * x);
            }

            forwardPropagation(neurons, inp_dats);

            for(int i = 0; i < N + 1; i++) {
                out_sin[idx    ][i] = inp_dats[i];
                out_sin[idx + 1][i] = neurons[neurons.size() - 1][i]->getX(); 
            }
            idx += 2;
        }
    }

    // 結果をファイルに出力する
    for(int i = 0; i < N + 1; i++) {
        ofs_sin << (2.0 * i / N - 1.0) << "\t";
        for(int j = 0; j < 2 * pattern_A * pattern_Omega; j++) {
            ofs_sin << out_sin[j][i] << "\t";
        }
        ofs_sin << endl;
    }

    // メモリの解放
    for(int i = 0; i < pattern_A * pattern_Omega; i++) {
        delete out_sin[i];
    }
    delete out_sin;
}

int main()
{
    srand((unsigned int)time(NULL));

    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    // ニューロンを5層でN+1-7-3-7-N+1にする
    neurons.resize(5);
    neurons[0].resize(N + 1);
    neurons[1].resize(7);
    neurons[2].resize(3);
    neurons[3].resize(7);
    neurons[4].resize(N + 1);

    vector<double> w(1);
    w[0] = 1.0;
    for(int i = 0; i < N + 1; i++) {
        neurons[0][i] = new Neuron(w, 0.0);
    }
    // ニューラルネットワークを構築する
    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i].size(); j++) {
            for(int k = 0; k < neurons[i - 1].size(); k++) {
                w[k] = my_rand(Rlow, Rhigh, 2);
            }
            neurons[i][j] = new Neuron(w, my_rand(Rlow, Rhigh, 2));
        }
    }

    outNetworkProperty("neuron_property.txt", neurons);


    // 教師データの作成
    vector<double> inp_dats[Patterns];
    vector<double> tsignal[Patterns];

    ofstream ofs_tsignal("tsignal.dat");
    ofs_tsignal << "# pattern\t" << "A\t" << "Omega\t" << endl;
    for(int i = 0; i < Patterns; i++) {
        double A = my_rand(0.1, 0.8, 2);
        double Omega = my_rand(OmegaLow, OmegaHigh, 2);
        for(int j = 0; j < N + 1; j++) {
            double inp_data = 2.0 * j / N - 1.0;
            double sin_data = A * sin(Omega * inp_data);
            inp_dats[i].push_back(sin_data);
            tsignal[i].push_back(sin_data);
        }
        ofs_tsignal << i << "\t" << A << "\t" << Omega << endl;
    }
    
    ofstream ofs_err("error.dat");
    ofs_err << "# " << "step\t" << "signal-error\t" << "estimate-error\t" << "general-error" << endl;
    ofstream ofs_w("out_w.dat");
    ofs_w << "# " << "step\t";
    for(int i = 1; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            const vector<double>& w = neurons[i][j]->getW();
            for(int k = 0; k < w.size(); k++) {
                ofs_w << "neurons[" << i << "][" << j << "]->w[" << k << "]\t";
            }
        }
    }
    ofs_w << endl;
    // 学習をする
    double sigError = calcSignalError(neurons, inp_dats, tsignal, Patterns);
    double geneError = calcGeneralError(neurons);
    int step = 0;
    for( ; sigError > ErrorEv && step < NUM_STEP; ) {
        for(int j = 0; j < 10; step++, j++) {
            if(step < 10) {
                stringstream filename;
                filename << "out_middle" << step << ".dat";
                outMiddleData(filename.str().c_str(), neurons);
                filename.str("");
                filename << "out_sin" << step << ".dat";
                outLearningSinData(filename.str().c_str(), neurons, inp_dats, tsignal);
                filename.str("");
                filename << "out_gene_sin" << step << ".dat";
                outGeneralSinData(filename.str().c_str(), neurons);
            }

            // ファイルに出力
            ofs_err  << step << "\t" << sigError << "\t" << 36 * sigError << "\t" << geneError << endl;
            cout << step << ", " << sigError << ", " << 36 * sigError << ", " << geneError << endl;
            ofs_w << step << "\t";
            for(int ii = 1; ii < neurons.size(); ii++) {
                for(int j = 0; j < neurons[ii].size(); j++) {
                    const vector<double>& w = neurons[ii][j]->getW();
                    for(int k = 0; k < w.size(); k++) {
                        ofs_w << w[k] << "\t";
                    }
                }
            }
            ofs_w << endl;

            for(int j = 0; j < Patterns; j++) {
                forwardPropagation(neurons, inp_dats[j]);
                backPropagation(neurons, tsignal[j]);
            }

            sigError  = calcSignalError(neurons, inp_dats, tsignal, Patterns);
            geneError = calcGeneralError(neurons);
        }
        stringstream filename;
        filename << "out_middle" << step << ".dat";
        outMiddleData(filename.str().c_str(), neurons);
        filename.str("");
        filename << "out_sin" << step << ".dat";
        outLearningSinData(filename.str().c_str(), neurons, inp_dats, tsignal);
        filename.str("");
        filename << "out_gene_sin" << step << ".dat";
        outGeneralSinData(filename.str().c_str(), neurons);
    }
    ofs_err.close();
    ofs_w.close();

// 深層学習
#if DEEP_LEARNING == 1
    // 深層学習用にファイルを作り直す
    ofs_err.open("error_deep.dat");
    ofs_err << "# " << "step\t" << "signal-error\t" << "estimate-error\t" << "general-error\t" << "Eta" << endl;
    ofs_w.open("out_w_deep.dat");
    ofs_w << "# " << "step\t";
    for(int i = 1; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            const vector<double>& w = neurons[i][j]->getW();
            for(int k = 0; k < w.size(); k++) {
                ofs_w << "neurons[" << i << "][" << j << "]->w[" << k << "]\t";
            }
        }
    }
    ofs_w << endl;

    Eta *= 0.5;     // Etaを半分に減らす
    // 最初の1step分を学習する
    for(int i = 0; i < DEEP_STEP - NUM_STEP; i++) {
        for(int j = 0; j < Patterns; j++) {
            forwardPropagation(neurons, inp_dats[j]);
            backPropagation(neurons, tsignal[j]);
        }
    }
    step = 1;
    sigError = calcSignalError(neurons, inp_dats, tsignal, Patterns);
    geneError = calcGeneralError(neurons);
    // 10stepまでやる
    for( ; step < 10; step++) {
        stringstream filename;
        filename << "out_middle_deep" << step << ".dat";
        outMiddleData(filename.str().c_str(), neurons);
        filename.str("");
        filename << "out_sin_deep" << step << ".dat";
        outLearningSinData(filename.str().c_str(), neurons, inp_dats, tsignal);
        filename.str("");
        filename << "out_gene_sin_deep" << step << ".dat";
        outGeneralSinData(filename.str().c_str(), neurons);
        // ファイルに出力
        ofs_err  << step << "\t" << sigError << "\t" << 36 * sigError << "\t" << geneError << "\t" << Eta << endl;
        cout << step << ", " << sigError << ", " << 36 * sigError << ", " << geneError << endl;
        ofs_w << step << "\t";
        for(int ii = 1; ii < neurons.size(); ii++) {
            for(int j = 0; j < neurons[ii].size(); j++) {
                const vector<double>& w = neurons[ii][j]->getW();
                for(int k = 0; k < w.size(); k++) {
                    ofs_w << w[k] << "\t";
                }
            }
        }
        ofs_w << endl;

        Eta *= 0.99;    // Etaを少しずつ減らしていく
        for(int i = 0; i < DEEP_STEP; i++) {
            for(int j = 0; j < Patterns; j++) {
                forwardPropagation(neurons, inp_dats[j]);
                backPropagation(neurons, tsignal[j]);
            }
        }

        sigError  = calcSignalError(neurons, inp_dats, tsignal, Patterns);
        geneError = calcGeneralError(neurons);
    }
    stringstream filename;
    filename << "out_middle_deep" << step << ".dat";
    outMiddleData(filename.str().c_str(), neurons);
    filename.str("");
    filename << "out_sin_deep" << step << ".dat";
    outLearningSinData(filename.str().c_str(), neurons, inp_dats, tsignal);
    filename.str("");
    filename << "out_gene_sin_deep" << step << ".dat";
    outGeneralSinData(filename.str().c_str(), neurons);

    for( ; sigError > ErrorEv && step < DEEP_NUM_STEP; ) {
        for(int j = 0; j < 10; step++, j++) {
            Eta *= 0.99;    // Etaを少しずつ減らしていく

            // ファイルに出力
            ofs_err  << step << "\t" << sigError << "\t" << 36 * sigError << "\t" << geneError << "\t" << Eta << endl;
            cout << step << ", " << sigError << ", " << 36 * sigError << ", " << geneError << endl;
            ofs_w << step << "\t";
            for(int ii = 1; ii < neurons.size(); ii++) {
                for(int j = 0; j < neurons[ii].size(); j++) {
                    const vector<double>& w = neurons[ii][j]->getW();
                    for(int k = 0; k < w.size(); k++) {
                        ofs_w << w[k] << "\t";
                    }
                }
            }
            ofs_w << endl;

            for(int i = 0; i < DEEP_STEP; i++) {
                for(int j = 0; j < Patterns; j++) {
                    forwardPropagation(neurons, inp_dats[j]);
                    backPropagation(neurons, tsignal[j]);
                }
            }

            sigError  = calcSignalError(neurons, inp_dats, tsignal, Patterns);
            geneError = calcGeneralError(neurons);
        }
        stringstream filename;
        filename << "out_middle_deep" << step << ".dat";
        outMiddleData(filename.str().c_str(), neurons);
        filename.str("");
        filename << "out_sin_deep" << step << ".dat";
        outLearningSinData(filename.str().c_str(), neurons, inp_dats, tsignal);
        filename.str("");
        filename << "out_gene_sin_deep" << step << ".dat";
        outGeneralSinData(filename.str().c_str(), neurons);
    }
#endif


    // ニューロンの削除（動的に確保したため）
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}