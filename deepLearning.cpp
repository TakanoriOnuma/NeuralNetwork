#include <time.h>

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;


const int    N       = 10;
const double Eta     = 0.5;
const double Alpha   = 0.8;
const double PAI     = 3.14159265359;
const double ErrorEv = 0.03;
const double Rlow    = -1.0;
const double Rhigh   = 1.0;

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

double calcError(vector<vector<Neuron*>>& neurons, const vector<double> inp_dats[], const vector<double> tsignal[], const int patterns)
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

void outNetworkProperty(const char* filename, const vector<vector<Neuron*>>& neurons)
{
    ofstream ofs(filename);

    ofs << "N:"       << N       << endl;
    ofs << "Eta:"     << Eta     << endl;
    ofs << "Alpha:"   << Alpha   << endl;
    ofs << "ErrorEv:" << ErrorEv << endl;
    ofs << "Rlow:"    << Rlow    << endl;
    ofs << "Rhigh:"   << Rhigh   << endl;
    ofs << "--- Neuron 階層:" << neurons.size() << " ---" << endl;
    for(int i = 0; i < neurons.size(); i++) {
        ofs << "第" << (i + 1) << "層:" << "ニューロン" << neurons[i].size() << "個" << endl;
    }
}

int main()
{
    srand((unsigned int)time(NULL));

    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    // ニューロンを3層でN+1-3-N+1にする
    neurons.resize(3);
    neurons[0].resize(N + 1);
    neurons[1].resize(3);
    neurons[2].resize(N + 1);

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
    const int Patterns = 30;
    vector<double> inp_dats[Patterns];
    vector<double> tsignal[Patterns];

    ofstream ofs_tsignal("tsignal.dat");
    double A[Patterns];
    double Lambda[Patterns];
    ofs_tsignal << "# pattern\t" << "A\t" << "Lambda\t" << endl;
    for(int i = 0; i < Patterns; i++) {
        A[i] = my_rand(-1.0, 1.0, 2);
        Lambda[i] = my_rand(0.1, PAI, 2);
        for(int j = 0; j < N + 1; j++) {
            double inp_data = 2.0 * j / N - 1.0;
            double sin_data = (A[i] * sin(Lambda[i] * inp_data) + A[i]) / (2 * A[i]);
            inp_dats[i].push_back(sin_data);
            tsignal[i].push_back(sin_data);
        }
        ofs_tsignal << i << "\t" << A[i] << "\t" << Lambda[i] << endl;
    }
    
    ofstream ofs_err("error.dat");
    ofs_err << "# " << "step\t" << "error" << endl;
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
    double vError = calcError(neurons, inp_dats, tsignal, Patterns);
    for(int i = 0; vError > ErrorEv && i < 1000; i++) {
        // ファイルに出力
        ofs_err << i << "\t" << vError << endl;
        ofs_w << i << "\t";
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
        vError = calcError(neurons, inp_dats, tsignal, Patterns);
    }

    // ニューロンデータの出力
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:"
                << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << ", "
                << neurons[i][j]->getU() << ", " << neurons[i][j]->getX() << endl;
      
            for each(double w in neurons[i][j]->getW()) {
                cout << w << ", ";
            }
            cout << endl;
            for each(double delta_w in neurons[i][j]->getDeltaW()) {
                cout << delta_w << ", ";
            }
            cout << endl;
        }
    }

    // 結果を出力
    ofstream ofs_sin("out_sin.dat");
    ofs_sin << "# ";
    for(int i = 0; i < inp_dats[0].size(); i++) {
        ofs_sin << "inp_dats[" << i << "]" << "\t";
    }
    for(int i = 0; i < tsignal[0].size(); i++) {
        ofs_sin << "tsignal[" << i << "]" << "\t";
    }
    for(int i = 0; i < neurons[neurons.size() - 1].size(); i++) {
        ofs_sin << "output[" << i << "]" << "\t";
    }
    ofs_sin << endl;

    ofstream ofs_x("out_x.dat");
    ofs_x << "# pattern" << "\t";
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            ofs_x << "neuron[" << i << "][" << j << "]" << "\t";
        }
    }
    ofs_x << endl;
    for(int i = 0; i < Patterns; i++) {
        for(int j = 0; j < inp_dats[i].size(); j++) {
            ofs_sin << inp_dats[i][j] << "\t";
        }
        forwardPropagation(neurons, inp_dats[i]);
        
        // 各ニューロンの出力をファイルに出力
        ofs_x << i << "\t";
        for(int j = 0; j < neurons.size(); j++) {
            for(int k = 0; k < neurons[j].size(); k++) {
                ofs_x << neurons[j][k]->getX() << "\t";
            }
        }
        ofs_x << endl;

        for(int j = 0; j < tsignal[i].size(); j++) {
            ofs_sin << (2 * A[i] * tsignal[i][j] - A[i]) << "\t";
        }
        for(int j = 0; j < neurons[neurons.size() - 1].size(); j++) {
            ofs_sin << (2 * A[i] * neurons[neurons.size() - 1][j]->getX() - A[i]) << "\t";
        }
        ofs_sin << endl;
    }

    // ニューロンの削除（動的に確保したため）
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}