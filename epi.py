import cv2
from ultralytics import YOLO
import math

# Carrega o modelo YOLO-World para detecção personalizada
model = YOLO("yolov8s-worldv2.pt")

# Classes que queremos detectar (ordem importa para os IDs!)
custom_classes = [
    "helmet", 
    "person", 
    "safety vest", 
    "safety goggles",      # Óculos de proteção
    "ear protection",      # Protetor auricular
    "earmuffs",           # Protetor auricular (alternativo)
    "bottle"
]
model.set_classes(custom_classes)

# IDs das classes (seguem a ordem da lista acima)
HELMET_ID = 0
PERSON_ID = 1
VEST_ID = 2
GOGGLES_ID = 3
EAR_PROTECTION_ID = 4
EARMUFFS_ID = 5
BOTTLE_ID = 6

# Cores
COLOR_SAFE = (0, 255, 0)        # Verde - Com todos os EPIs
COLOR_PARTIAL = (0, 255, 255)   # Amarelo - Com alguns EPIs
COLOR_UNSAFE = (0, 0, 255)      # Vermelho - Sem EPIs
COLOR_HELMET = (0, 255, 255)    # Amarelo - Capacete
COLOR_VEST = (0, 165, 255)      # Laranja - Colete
COLOR_GOGGLES = (255, 0, 255)   # Magenta - Óculos
COLOR_EAR = (255, 100, 255)     # Rosa - Protetor auricular
COLOR_BOTTLE = (255, 255, 0)    # Ciano - Garrafa

def check_ppe_on_person(person_box, ppe_boxes):
    """
    Verifica se há um EPI próximo à região da cabeça da pessoa.
    A cabeça é considerada como 40% superior da caixa da pessoa.
    """
    px1, py1, px2, py2 = person_box
    
    # Região da cabeça (40% superior da pessoa)
    head_y1 = py1
    head_y2 = py1 + (py2 - py1) * 0.4
    head_x1 = px1
    head_x2 = px2
    
    # Verifica se algum EPI está na região da cabeça
    for ex1, ey1, ex2, ey2 in ppe_boxes:
        # Calcula o centro do EPI
        ppe_center_x = (ex1 + ex2) / 2
        ppe_center_y = (ey1 + ey2) / 2
        
        # Verifica se o centro do EPI está na região da cabeça
        if (head_x1 <= ppe_center_x <= head_x2 and 
            head_y1 <= ppe_center_y <= head_y2):
            return True
    
    return False


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("=" * 70)
print("SISTEMA DE MONITORAMENTO DE EPI - VERSÃO COMPLETA")
print("=" * 70)
print("EPIs Monitorados:")
print("  🪖 Capacete")
print("  🥽 Óculos de Proteção")
print("  🎧 Protetor Auricular")
print("  🦺 Colete Refletivo")
print("\nStatus de Conformidade:")
print("  ✓ Verde   = Todos os EPIs detectados (SEGURO)")
print("  ⚠ Amarelo = Alguns EPIs faltando (ATENÇÃO)")
print("  ✗ Vermelho = EPIs críticos faltando (PERIGO)")
print("\nPressione 'q' para sair.")
print("=" * 70)

while True:
    success, img = cap.read()
    if not success:
        break

    # Realiza a detecção
    results = model.predict(img, conf=0.20, verbose=False)

    # Separa as detecções por tipo
    persons = []
    helmets = []
    vests = []
    goggles = []
    ear_protections = []
    bottles = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id == PERSON_ID:
                persons.append((x1, y1, x2, y2, conf))
            elif cls_id == HELMET_ID:
                helmets.append((x1, y1, x2, y2, conf))
            elif cls_id == VEST_ID:
                vests.append((x1, y1, x2, y2, conf))
            elif cls_id == GOGGLES_ID:
                goggles.append((x1, y1, x2, y2, conf))
            elif cls_id == EAR_PROTECTION_ID or cls_id == EARMUFFS_ID:
                ear_protections.append((x1, y1, x2, y2, conf))
            elif cls_id == BOTTLE_ID:
                bottles.append((x1, y1, x2, y2, conf))
    
    # Desenha os EPIs detectados
    for hx1, hy1, hx2, hy2, conf in helmets:
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), COLOR_HELMET, 2)
        cv2.putText(img, f'Capacete {conf:.2f}', (hx1, hy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HELMET, 2)
    
    for gx1, gy1, gx2, gy2, conf in goggles:
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), COLOR_GOGGLES, 2)
        cv2.putText(img, f'Oculos {conf:.2f}', (gx1, gy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GOGGLES, 2)
    
    for ex1, ey1, ex2, ey2, conf in ear_protections:
        cv2.rectangle(img, (ex1, ey1), (ex2, ey2), COLOR_EAR, 2)
        cv2.putText(img, f'Prot.Auricular {conf:.2f}', (ex1, ey1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EAR, 2)
    
    for vx1, vy1, vx2, vy2, conf in vests:
        cv2.rectangle(img, (vx1, vy1), (vx2, vy2), COLOR_VEST, 2)
        cv2.putText(img, f'Colete {conf:.2f}', (vx1, vy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VEST, 2)
    
    for bx1, by1, bx2, by2, conf in bottles:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), COLOR_BOTTLE, 2)
        cv2.putText(img, f'Garrafa {conf:.2f}', (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOTTLE, 2)
    
    # Analisa cada pessoa detectada
    helmet_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in helmets]
    goggles_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in goggles]
    ear_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in ear_protections]
    
    for px1, py1, px2, py2, conf in persons:
        # Verifica cada EPI
        has_helmet = check_ppe_on_person((px1, py1, px2, py2), helmet_boxes)
        has_goggles = check_ppe_on_person((px1, py1, px2, py2), goggles_boxes)
        has_ear_protection = check_ppe_on_person((px1, py1, px2, py2), ear_boxes)
        
        # Conta quantos EPIs a pessoa está usando
        ppe_count = sum([has_helmet, has_goggles, has_ear_protection])
        
        # Define status e cor
        status_lines = []
        if has_helmet:
            status_lines.append("✓ Capacete")
        else:
            status_lines.append("✗ Capacete")
            
        if has_goggles:
            status_lines.append("✓ Oculos")
        else:
            status_lines.append("✗ Oculos")
            
        if has_ear_protection:
            status_lines.append("✓ Prot.Auric")
        else:
            status_lines.append("✗ Prot.Auric")
        
        # Define cor baseada na conformidade
        if ppe_count == 3:
            color = COLOR_SAFE
            main_status = "SEGURO"
        elif ppe_count >= 1:
            color = COLOR_PARTIAL
            main_status = "ATENCAO"
        else:
            color = COLOR_UNSAFE
            main_status = "PERIGO"
        
        # Desenha retângulo ao redor da pessoa
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 4)
        
        # Desenha painel de status
        panel_y = py1 - 100
        if panel_y < 0:
            panel_y = py2 + 10
        
        # Fundo do painel
        cv2.rectangle(img, (px1, panel_y), (px1 + 200, panel_y + 90), (0, 0, 0), -1)
        cv2.rectangle(img, (px1, panel_y), (px1 + 200, panel_y + 90), color, 2)
        
        # Texto do status
        cv2.putText(img, f"Status: {main_status}", (px1 + 5, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = 40
        for status_line in status_lines:
            text_color = COLOR_SAFE if "✓" in status_line else COLOR_UNSAFE
            cv2.putText(img, status_line, (px1 + 5, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
            y_offset += 18
    
    # Adiciona contador no canto superior
    info_bg_height = 80
    cv2.rectangle(img, (0, 0), (400, info_bg_height), (0, 0, 0), -1)
    
    cv2.putText(img, f'Pessoas: {len(persons)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f'Capacetes: {len(helmets)} | Oculos: {len(goggles)}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f'Prot.Auricular: {len(ear_protections)}', (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Sistema de Monitoramento EPI", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSistema encerrado.")
# postei no github