import { Component, ChangeDetectorRef, ChangeDetectionStrategy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'home',
  imports: [CommonModule, FormsModule],
  templateUrl: './home.html',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class Home {
  aiScore: number | null = null;
  textContent: string = '';
  selectedFile: File | null = null;
  loading: boolean = false;
  statusMessage: string = 'Awaiting input...';

  private readonly API_BASE = 'http://127.0.0.1:8001/api/ai';

  constructor(private cdr: ChangeDetectorRef, private http: HttpClient) {
    this.triggerCDR('Component Initialized');
  }

  onInput(type: string): void {
    this.statusMessage = `${type} input detected. Ready for analysis.`;
    this.triggerCDR(`Input registered for ${type}`);
  }

  onFileSelected(event: any, type: string): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.statusMessage = `${type.charAt(0).toUpperCase() + type.slice(1)} file "${file.name}" selected.`;
      this.triggerCDR(`File selected: ${file.name}`);
    }
  }

  analyzeText(): void {
    if (!this.textContent.trim()) return;
    this.setLoadingState('Analyzing text via DeBERTa-v3...');

    this.http.post<any>(`${this.API_BASE}/text/detect`, { text: this.textContent })
      .subscribe({
        next: (res) => this.handleSuccess(res.aiScore, 'Text analysis complete'),
        error: (err) => this.handleError(err, 'Text analysis failed')
      });
  }

  analyzeFile(endpoint: string): void {
    if (!this.selectedFile) return;
    this.setLoadingState(`Extracting and analyzing ${endpoint.split('/')[0]} payload...`);

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post<any>(`${this.API_BASE}/${endpoint}`, formData)
      .subscribe({
        next: (res) => this.handleSuccess(res.aiScore, `${endpoint.split('/')[0]} analysis complete`),
        error: (err) => this.handleError(err, `${endpoint.split('/')[0]} analysis failed`)
      });
  }

  private setLoadingState(msg: string): void {
    this.loading = true;
    this.aiScore = null;
    this.statusMessage = msg;
    this.triggerCDR('Pipeline loading state initiated');
  }

  private handleSuccess(score: number, msg: string): void {
    this.aiScore = score;
    this.loading = false;
    this.statusMessage = msg;
    this.triggerCDR(`Pipeline success: Score ${score}%`);
  }

  private handleError(error: any, context: string): void {
    this.loading = false;
    this.aiScore = null;
    this.statusMessage = `Error: ${error.message || 'API connection refused'}`;
    this.triggerCDR(`Pipeline error: ${context}`);
  }

  getScoreColor(): string {
    if (this.aiScore === null) return 'text-gray-700';
    if (this.aiScore > 75) return 'text-red-500 drop-shadow-[0_0_20px_rgba(239,68,68,0.9)]';
    if (this.aiScore > 40) return 'text-yellow-400 drop-shadow-[0_0_20px_rgba(250,204,21,0.9)]';
    return 'text-green-500 drop-shadow-[0_0_20px_rgba(34,197,94,0.9)]';
  }

  private triggerCDR(stepName: string): void {
    console.log(`[CDR Triggered] Transition: ${stepName}`);
    this.cdr.markForCheck();
  }
}
